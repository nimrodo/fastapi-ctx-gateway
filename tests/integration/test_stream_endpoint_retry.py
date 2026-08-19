"""Router-level test: a pre-stream Gemini failure is retried once, end to end."""

import httpx
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    return Settings()


def test_pre_stream_failure_is_retried_once_then_succeeds(monkeypatch) -> None:
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},'
        b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":3}}\n\n'
    )
    responses = [
        httpx.Response(500, content=b"internal error"),
        httpx.Response(200, content=sse_body, headers={"content-type": "text/event-stream"}),
    ]

    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent")
        route.side_effect = responses

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert route.call_count == 2
    assert response.status_code == 200
    assert response.content == sse_body


def test_pre_stream_failure_exhausts_retry_and_returns_terminal_error(monkeypatch) -> None:
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert route.call_count == 2  # one initial attempt + one retry, no more
    assert response.status_code == 200  # SSE stream started; the error is *inside* the body
    assert b'"error"' in response.content
