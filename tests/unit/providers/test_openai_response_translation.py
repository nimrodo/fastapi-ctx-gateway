"""Tests for OpenAIProvider's OpenAI-native SSE -> neutral SSE translation."""

import json

import httpx
import respx

from fastapi_ctx_gateway.providers.openai import OpenAIProvider, _translate_sse
from fastapi_ctx_gateway.schemas.neutral import NeutralGenerateRequest, TextPart, Turn

OPENAI_EVENT_1 = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},'
    b'"finish_reason":null}]}\n\n'
)
OPENAI_EVENT_2 = (
    b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
)
OPENAI_USAGE_EVENT = (
    b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
)
OPENAI_DONE = b"data: [DONE]\n\n"


async def _upstream(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


def _payload(chunk: bytes) -> dict:
    return json.loads(chunk.removeprefix(b"data: ").strip())


async def test_translate_sse_yields_one_neutral_event_per_native_event_and_skips_done() -> None:
    chunks = [
        c
        async for c in _translate_sse(
            _upstream([OPENAI_EVENT_1, OPENAI_EVENT_2, OPENAI_USAGE_EVENT, OPENAI_DONE])
        )
    ]
    assert len(chunks) == 3
    first = _payload(chunks[0])
    assert first["delta"]["parts"][0]["text"] == "Hel"
    assert first.get("finish_reason") is None
    second = _payload(chunks[1])
    assert second["delta"]["parts"][0]["text"] == "lo"
    assert second["finish_reason"] == "stop"
    third = _payload(chunks[2])
    assert third["usage"]["total_tokens"] == 5
    assert third["usage"]["prompt_tokens"] == 2
    assert third["usage"]["completion_tokens"] == 3


async def test_stream_success_end_to_end_sends_correct_request_and_translates_response() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    with respx.mock(base_url="https://api.openai.com") as mock:
        route = mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                content=OPENAI_EVENT_1 + OPENAI_EVENT_2 + OPENAI_DONE,
                headers={"content-type": "text/event-stream"},
            )
        )
        async with httpx.AsyncClient() as http_client:
            provider = OpenAIProvider(
                http_client=http_client, api_key="sk-test", base_url="https://api.openai.com/v1"
            )
            chunks = [c async for c in provider.stream("gpt-4o", request)]

    assert len(chunks) == 2
    sent = route.calls[0].request
    assert sent.headers["authorization"] == "Bearer sk-test"


async def test_stream_yields_neutral_error_event_on_non_2xx_response() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )
        async with httpx.AsyncClient() as http_client:
            provider = OpenAIProvider(
                http_client=http_client, api_key="sk-test", base_url="https://api.openai.com/v1"
            )
            chunks = [c async for c in provider.stream("gpt-4o", request)]

    assert len(chunks) == 1
    payload = _payload(chunks[0])
    assert payload["error"]["provider_status"] == 500


async def test_stream_yields_neutral_error_event_on_transport_failure() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    with respx.mock(base_url="https://api.openai.com") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=httpx.ConnectError("boom"))
        async with httpx.AsyncClient() as http_client:
            provider = OpenAIProvider(
                http_client=http_client, api_key="sk-test", base_url="https://api.openai.com/v1"
            )
            chunks = [c async for c in provider.stream("gpt-4o", request)]

    assert len(chunks) == 1
    payload = _payload(chunks[0])
    assert payload["error"]["provider_status"] is None
