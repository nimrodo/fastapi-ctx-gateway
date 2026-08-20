"""Shared low-level SSE framing and error-event helpers, used by every provider adapter.

Each provider's own response translator still owns interpreting the JSON
payload of an event — only the "how do we split raw upstream bytes into
per-event `data:` payloads, buffering at most one event at a time" and "how
do we build the terminal neutral error event" pieces are shared, since both
are identical regardless of which upstream API is being called.
"""

from collections.abc import AsyncIterator

from fastapi_ctx_gateway.schemas.neutral import NeutralError, NeutralErrorEvent

__all__ = ["iter_sse_data_lines", "neutral_error_event"]


async def iter_sse_data_lines(native_bytes: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    r"""Buffer only up to the next SSE event boundary and yield each event's `data:` payload.

    Never buffers more than one event's worth of bytes at a time — this is
    what keeps upstream-event -> neutral-event translation 1:1 without
    adding more than one event's worth of latency. A trailing partial
    buffer (no closing `\n\n`) has nothing to yield.
    """
    buffer = b""
    async for chunk in native_bytes:
        buffer += chunk
        while b"\n\n" in buffer:
            raw_event, buffer = buffer.split(b"\n\n", 1)
            for line in raw_event.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:") :].strip()
                if payload:
                    yield payload
                break


def neutral_error_event(
    message: str, status: int | None, error_type: str = "upstream_error"
) -> bytes:
    """Build the terminal neutral SSE error event every provider yields on failure."""
    payload = NeutralErrorEvent(
        error=NeutralError(message=message, type=error_type, provider_status=status)
    )
    return f"data: {payload.model_dump_json()}\n\n".encode()
