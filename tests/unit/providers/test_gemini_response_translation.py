"""Tests for GeminiProvider's Gemini-native SSE -> neutral SSE translation."""

import json

import httpx
import respx

from fastapi_ctx_gateway.providers.gemini import GeminiProvider, _translate_sse
from fastapi_ctx_gateway.schemas.neutral import NeutralGenerateRequest, TextPart, Turn

GEMINI_EVENT_1 = b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"}}]}\n\n'
GEMINI_EVENT_2 = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}],"role":"model"},'
    b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":5,"promptTokenCount":2,'
    b'"candidatesTokenCount":3}}\n\n'
)


async def _upstream(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


def _payload(chunk: bytes) -> dict:
    return json.loads(chunk.removeprefix(b"data: ").strip())


async def test_translate_sse_yields_one_neutral_event_per_native_event() -> None:
    chunks = [c async for c in _translate_sse(_upstream([GEMINI_EVENT_1, GEMINI_EVENT_2]))]
    assert len(chunks) == 2
    first = _payload(chunks[0])
    assert first["delta"]["parts"][0]["text"] == "Hel"
    assert first.get("finish_reason") is None
    second = _payload(chunks[1])
    assert second["delta"]["parts"][0]["text"] == "lo"
    assert second["finish_reason"] == "stop"
    assert second["usage"]["total_tokens"] == 5
    assert second["usage"]["prompt_tokens"] == 2
    assert second["usage"]["completion_tokens"] == 3


async def test_translate_sse_preserves_one_to_one_boundary_when_bytes_split_mid_event() -> None:
    combined = GEMINI_EVENT_1 + GEMINI_EVENT_2
    midpoint = len(combined) // 2
    chunks = [
        c async for c in _translate_sse(_upstream([combined[:midpoint], combined[midpoint:]]))
    ]
    assert len(chunks) == 2


async def test_stream_success_end_to_end_sends_correct_request_and_translates_response() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200,
                content=GEMINI_EVENT_1 + GEMINI_EVENT_2,
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient() as http_client:
            provider = GeminiProvider(
                http_client=http_client,
                api_key="upstream-key",
                base_url="https://generativelanguage.googleapis.com",
            )
            chunks = [c async for c in provider.stream("gemini-3.7-flash", request)]

    assert len(chunks) == 2
    sent = route.calls[0].request
    assert sent.headers["x-goog-api-key"] == "upstream-key"
    assert sent.url.params["alt"] == "sse"


async def test_stream_yields_neutral_error_event_on_non_2xx_response() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )
        async with httpx.AsyncClient() as http_client:
            provider = GeminiProvider(
                http_client=http_client,
                api_key="k",
                base_url="https://generativelanguage.googleapis.com",
            )
            chunks = [c async for c in provider.stream("gemini-3.7-flash", request)]

    assert len(chunks) == 1
    payload = _payload(chunks[0])
    assert payload["error"]["provider_status"] == 500


async def test_stream_error_message_extracted_from_json_error_envelope() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    error_body = json.dumps({"error": {"message": "API key not valid", "code": 400}}).encode()
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(400, content=error_body)
        )
        async with httpx.AsyncClient() as http_client:
            provider = GeminiProvider(
                http_client=http_client,
                api_key="k",
                base_url="https://generativelanguage.googleapis.com",
            )
            chunks = [c async for c in provider.stream("gemini-3.7-flash", request)]

    payload = _payload(chunks[0])
    assert payload["error"]["message"] == "Gemini returned 400: API key not valid"


async def test_stream_yields_neutral_error_event_on_transport_failure() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            side_effect=httpx.ConnectError("boom")
        )
        async with httpx.AsyncClient() as http_client:
            provider = GeminiProvider(
                http_client=http_client,
                api_key="k",
                base_url="https://generativelanguage.googleapis.com",
            )
            chunks = [c async for c in provider.stream("gemini-3.7-flash", request)]

    assert len(chunks) == 1
    payload = _payload(chunks[0])
    assert payload["error"]["provider_status"] is None
