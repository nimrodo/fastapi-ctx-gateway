"""OpenAI provider adapter: neutral contract <-> OpenAI's Chat Completions streaming API."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.providers.sse import iter_sse_data_lines, neutral_error_event
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    Delta,
    FinishReason,
    NeutralGenerateRequest,
    NeutralStreamEvent,
    Part,
    TextPart,
    Usage,
)

__all__ = ["OpenAIProvider"]

_DONE_SENTINEL = b"[DONE]"

_FINISH_REASON_FROM_OPENAI = {
    "stop": FinishReason.STOP,
    "length": FinishReason.MAX_TOKENS,
    "content_filter": FinishReason.SAFETY,
}


class OpenAIProvider(Provider):
    """Translates the neutral contract to/from OpenAI's Chat Completions streaming API."""

    name = "openai"

    # Matches GeminiProvider's bounded pre-stream retry: one retry, only
    # for a failure before any bytes reached the client.
    _MAX_RETRIES = 1

    def __init__(self, http_client: httpx.AsyncClient, api_key: str, base_url: str) -> None:
        """Wrap a shared client with the credentials/base URL for one deployment."""
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def stream(self, model: str, request: NeutralGenerateRequest) -> AsyncIterator[bytes]:
        """Call OpenAI and yield neutral SSE bytes, one native event per yielded chunk."""
        native_body = _to_openai_request(model, request)
        url = f"{self._base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "content-type": "application/json"}
        attempts = 0
        while True:
            attempts += 1
            yielded_any = False
            try:
                async with self._http_client.stream(
                    "POST", url, headers=headers, json=native_body
                ) as response:
                    if response.is_error:
                        if attempts <= self._MAX_RETRIES:
                            continue
                        body = await response.aread()
                        yield neutral_error_event(
                            _status_error_message(response.status_code, body),
                            response.status_code,
                        )
                        return
                    async for neutral_chunk in _translate_sse(response.aiter_bytes()):
                        yielded_any = True
                        yield neutral_chunk
                    return
            except httpx.HTTPError as exc:
                if not yielded_any and attempts <= self._MAX_RETRIES:
                    continue
                yield neutral_error_event(str(exc), None)
                return


def _status_error_message(status_code: int, body: bytes) -> str:
    text = body.decode(errors="replace").strip()
    return f"OpenAI returned {status_code}: {text}" if text else f"OpenAI returned {status_code}"


# --- request translation: neutral -> OpenAI native ---


def _part_to_openai_content(part: TextPart | BinaryPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    url = f"data:{part.mime_type};base64,{part.data}" if part.data is not None else part.uri
    return {"type": "image_url", "image_url": {"url": url}}


def _turn_to_openai_message(role: str, parts: list[Part]) -> dict[str, Any]:
    return {"role": role, "content": [_part_to_openai_content(p) for p in parts]}


def _to_openai_request(model: str, request: NeutralGenerateRequest) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append(_turn_to_openai_message("system", request.system))
    messages.extend(_turn_to_openai_message(turn.role, turn.parts) for turn in request.turns)

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.tools:
        body["tools"] = request.tools
    if request.tool_config:
        body["tool_choice"] = request.tool_config
    # safety_settings has no OpenAI equivalent and is intentionally dropped.

    config = request.generation_config
    if config is not None:
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.top_p is not None:
            body["top_p"] = config.top_p
        if config.max_output_tokens is not None:
            body["max_tokens"] = config.max_output_tokens
        if config.stop_sequences is not None:
            body["stop"] = config.stop_sequences
        if config.candidate_count is not None:
            body["n"] = config.candidate_count
        # top_k has no OpenAI equivalent and is intentionally dropped.
    return body


# --- response translation: OpenAI native SSE -> neutral SSE, one event -> one event ---


async def _translate_sse(native_bytes: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Translate each OpenAI-native SSE event to exactly one neutral SSE event.

    The `[DONE]` sentinel line (not JSON) is OpenAI's own stream terminator
    and carries no content of its own — it's swallowed here rather than
    translated to an (empty) neutral event.
    """
    async for payload in iter_sse_data_lines(native_bytes):
        if payload == _DONE_SENTINEL:
            continue
        neutral_chunk = _translate_one_event(payload)
        if neutral_chunk is not None:
            yield neutral_chunk


def _translate_one_event(payload: bytes) -> bytes | None:
    try:
        event_json = json.loads(payload)
    except ValueError:
        return None
    neutral_event = _openai_chunk_to_neutral(event_json)
    return f"data: {neutral_event.model_dump_json(exclude_none=True)}\n\n".encode()


def _openai_chunk_to_neutral(event_json: dict[str, Any]) -> NeutralStreamEvent:
    usage_json = event_json.get("usage")
    usage = (
        Usage(
            prompt_tokens=usage_json.get("prompt_tokens"),
            completion_tokens=usage_json.get("completion_tokens"),
            total_tokens=usage_json.get("total_tokens"),
        )
        if usage_json
        else None
    )
    choices = event_json.get("choices") or []
    if not choices:
        return NeutralStreamEvent(usage=usage)
    choice = choices[0]
    native_finish_reason = choice.get("finish_reason")
    finish_reason = (
        _FINISH_REASON_FROM_OPENAI.get(native_finish_reason, FinishReason.OTHER)
        if native_finish_reason
        else None
    )
    content = (choice.get("delta") or {}).get("content")
    parts: list[Part] = [TextPart(text=content)] if content else []
    delta = Delta(role="assistant", parts=parts)
    return NeutralStreamEvent(delta=delta, finish_reason=finish_reason, usage=usage)
