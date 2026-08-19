"""Tees a neutral SSE byte stream to the client while accumulating text."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from fastapi_ctx_gateway.schemas.neutral import NeutralError, NeutralErrorEvent, Usage

__all__ = ["StreamAccumulator", "stream_with_retry", "tee_stream"]


class StreamAccumulator(BaseModel):
    """Side-channel state populated as tee_stream forwards chunks.

    Passed in by the caller (rather than returned) because an async
    generator's return value isn't observable mid-stream — this lets the
    caller inspect accumulated state once iteration finishes.
    """

    text: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    bytes_streamed: bool = False
    errored: bool = False

    @property
    def finished_cleanly(self) -> bool:
        """True once a terminal finish_reason chunk has been observed, with no error."""
        return self.finish_reason is not None and not self.errored


def _extract_delta(
    event_json: dict[str, Any],
) -> tuple[str, str | None, Usage | None, bool]:
    if "error" in event_json:
        return "", None, None, True
    delta = event_json.get("delta") or {}
    text = "".join(
        part.get("text", "") for part in delta.get("parts", []) if part.get("type") == "text"
    )
    usage_json = event_json.get("usage")
    usage = Usage.model_validate(usage_json) if usage_json else None
    return text, event_json.get("finish_reason"), usage, False


async def tee_stream(
    upstream: AsyncIterator[bytes], accumulator: StreamAccumulator
) -> AsyncIterator[bytes]:
    """Forward each raw chunk to the caller while accumulating text/finish_reason.

    Forwarding happens before parsing, so upstream bytes reach the client
    untouched even if a chunk can't be interpreted as SSE (e.g. an error
    body). If the caller stops iterating early (client disconnect), the
    upstream async generator is explicitly closed so the in-flight upstream
    call is cancelled rather than drained to completion.
    """
    buffer = b""
    try:
        async for chunk in upstream:
            accumulator.bytes_streamed = True
            yield chunk
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[len(b"data:") :].strip()
                    if not payload:
                        continue
                    try:
                        event_json = json.loads(payload)
                    except ValueError:
                        continue
                    text, finish_reason, usage, errored = _extract_delta(event_json)
                    accumulator.text += text
                    if finish_reason:
                        accumulator.finish_reason = finish_reason
                    if usage:
                        accumulator.usage = usage
                    if errored:
                        accumulator.errored = True
    finally:
        aclose = getattr(upstream, "aclose", None)
        if aclose is not None:
            await aclose()


def _terminal_error_event(message: str) -> bytes:
    payload = NeutralErrorEvent(error=NeutralError(message=message, type="stream_failure"))
    return f"data: {payload.model_dump_json()}\n\n".encode()


async def stream_with_retry(
    open_stream: Callable[[], AsyncIterator[bytes]],
    accumulator: StreamAccumulator,
    max_retries: int = 1,
) -> AsyncIterator[bytes]:
    """Open a stream, retrying once (no backoff) only if it fails before any bytes are sent.

    A failure after bytes have already reached the client is never
    retried — the client may have partially rendered a response, and
    starting over risks duplicate or corrupted output. Either way, once
    retries are exhausted (or immediately, if bytes already streamed), a
    terminal SSE error event is yielded so the client gets a clean signal
    rather than an abruptly closed connection. This function never raises.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            async for chunk in tee_stream(open_stream(), accumulator):
                yield chunk
            return
        except Exception as exc:
            if not accumulator.bytes_streamed and attempts <= max_retries:
                continue
            yield _terminal_error_event(str(exc))
            return
