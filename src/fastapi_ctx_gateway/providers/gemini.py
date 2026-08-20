"""Gemini provider adapter: neutral contract <-> Gemini's own streamGenerateContent wire format."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.providers.gemini_wire import (
    Content as GeminiContent,
)
from fastapi_ctx_gateway.providers.gemini_wire import (
    GenerateContentRequest as GeminiRequest,
)
from fastapi_ctx_gateway.providers.gemini_wire import (
    GenerationConfig as GeminiGenerationConfig,
)
from fastapi_ctx_gateway.providers.gemini_wire import (
    Part as GeminiPart,
)
from fastapi_ctx_gateway.providers.sse import iter_sse_data_lines, neutral_error_event
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    Delta,
    FinishReason,
    NeutralGenerateRequest,
    NeutralStreamEvent,
    Part,
    TextPart,
    Turn,
    Usage,
)

__all__ = ["GeminiProvider"]

_ROLE_TO_GEMINI = {"user": "user", "assistant": "model"}
_FINISH_REASON_FROM_GEMINI = {
    "STOP": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.MAX_TOKENS,
    "SAFETY": FinishReason.SAFETY,
}


class GeminiProvider(Provider):
    """Translates the neutral contract to/from Gemini's streamGenerateContent API."""

    name = "gemini"

    # Matches the bounded pre-stream retry the gateway has always offered:
    # one retry, only for a failure before any bytes reached the client.
    _MAX_RETRIES = 1

    def __init__(self, http_client: httpx.AsyncClient, api_key: str, base_url: str) -> None:
        """Wrap a shared client with the credentials/base URL for one deployment."""
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def stream(self, model: str, request: NeutralGenerateRequest) -> AsyncIterator[bytes]:
        """Call Gemini and yield neutral SSE bytes, one native event per yielded chunk."""
        native_body = _to_gemini_request(request).model_dump(by_alias=True, exclude_none=True)
        url = f"{self._base_url}/v1beta/models/{model}:streamGenerateContent"
        headers = {"x-goog-api-key": self._api_key, "content-type": "application/json"}
        attempts = 0
        while True:
            attempts += 1
            yielded_any = False
            try:
                async with self._http_client.stream(
                    "POST", url, params={"alt": "sse"}, headers=headers, json=native_body
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
    return f"Gemini returned {status_code}: {text}" if text else f"Gemini returned {status_code}"


# --- request translation: neutral -> Gemini native ---


def _part_to_gemini(part: TextPart | BinaryPart) -> GeminiPart:
    if isinstance(part, TextPart):
        return GeminiPart(text=part.text)
    if part.data is not None:
        return GeminiPart(inline_data={"mimeType": part.mime_type, "data": part.data})
    return GeminiPart(file_data={"mimeType": part.mime_type, "fileUri": part.uri})


def _turn_to_gemini(turn: Turn) -> GeminiContent:
    return GeminiContent(
        role=_ROLE_TO_GEMINI[turn.role], parts=[_part_to_gemini(p) for p in turn.parts]
    )


def _to_gemini_request(request: NeutralGenerateRequest) -> GeminiRequest:
    system_instruction = None
    if request.system:
        system_instruction = GeminiContent(
            role="system", parts=[_part_to_gemini(p) for p in request.system]
        )
    generation_config = None
    if request.generation_config is not None:
        generation_config = GeminiGenerationConfig(
            **request.generation_config.model_dump(exclude_none=True)
        )
    return GeminiRequest(
        contents=[_turn_to_gemini(turn) for turn in request.turns],
        tools=request.tools,
        tool_config=request.tool_config,
        safety_settings=request.safety_settings,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )


# --- response translation: Gemini native SSE -> neutral SSE, one event -> one event ---


async def _translate_sse(native_bytes: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Translate each Gemini-native SSE event to exactly one neutral SSE event."""
    async for payload in iter_sse_data_lines(native_bytes):
        neutral_chunk = _translate_one_event(payload)
        if neutral_chunk is not None:
            yield neutral_chunk


def _translate_one_event(payload: bytes) -> bytes | None:
    try:
        event_json = json.loads(payload)
    except ValueError:
        return None
    neutral_event = _gemini_delta_to_neutral(event_json)
    return f"data: {neutral_event.model_dump_json(exclude_none=True)}\n\n".encode()


def _gemini_delta_to_neutral(event_json: dict[str, Any]) -> NeutralStreamEvent:
    candidates = event_json.get("candidates") or []
    usage_json = event_json.get("usageMetadata")
    usage = (
        Usage(
            prompt_tokens=usage_json.get("promptTokenCount"),
            completion_tokens=usage_json.get("candidatesTokenCount"),
            total_tokens=usage_json.get("totalTokenCount"),
        )
        if usage_json
        else None
    )
    if not candidates:
        return NeutralStreamEvent(usage=usage)
    candidate = candidates[0]
    native_finish_reason = candidate.get("finishReason")
    finish_reason = (
        _FINISH_REASON_FROM_GEMINI.get(native_finish_reason, FinishReason.OTHER)
        if native_finish_reason
        else None
    )
    native_parts = (candidate.get("content") or {}).get("parts") or []
    parts: list[Part] = [
        TextPart(text=native_part["text"]) for native_part in native_parts if "text" in native_part
    ]
    delta = Delta(role="assistant", parts=parts)
    return NeutralStreamEvent(delta=delta, finish_reason=finish_reason, usage=usage)
