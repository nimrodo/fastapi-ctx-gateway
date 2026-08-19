"""Tests for GeminiClient.stream_generate."""

import httpx
import respx

from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.schemas.gemini import Content, GenerateContentRequest, Part

SSE_BODY = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"}}]}\n\n'
    b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}],"role":"model"},'
    b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":5}}\n\n'
)


async def test_stream_generate_sends_correct_request_and_yields_chunks() -> None:
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        )

        async with httpx.AsyncClient() as http_client:
            client = GeminiClient(
                http_client=http_client,
                api_key="upstream-key",
                base_url="https://generativelanguage.googleapis.com",
            )
            request = GenerateContentRequest(
                contents=[Content(role="user", parts=[Part(text="hi")])]
            )
            chunks = [chunk async for chunk in client.stream_generate("gemini-2.5-flash", request)]

        assert b"".join(chunks) == SSE_BODY
        sent = route.calls[0].request
        assert sent.headers["x-goog-api-key"] == "upstream-key"
        assert sent.url.params["alt"] == "sse"
