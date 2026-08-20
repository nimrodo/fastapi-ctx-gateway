"""Router-level test: the OpenAI provider works end to end through the same route shape."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

pytestmark = pytest.mark.integration

OPENAI_SSE_BODY = (
    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},'
    b'"finish_reason":null}]}\n\n'
    b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}\n\n'
    b"data: [DONE]\n\n"
)


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    monkeypatch.setenv("GATEWAY_OPENAI_API_KEY", "sk-test")
    return Settings()


def test_openai_route_streams_translated_neutral_body(monkeypatch) -> None:
    with respx.mock(base_url="https://api.openai.com") as mock:
        route = mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=OPENAI_SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            response = client.post(
                "/v1/openai/gpt-4o:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
            )

    assert response.status_code == 200
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
    assert b'"text":"Hel"' in response.content
    assert b'"finish_reason":"stop"' in response.content
    assert b'"total_tokens":5' in response.content
    # The [DONE] sentinel must never leak through as a translated event.
    assert b"[DONE]" not in response.content


def test_openai_route_404s_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    app = create_app(Settings())  # no GATEWAY_OPENAI_API_KEY set
    with TestClient(app) as client:
        response = client.post(
            "/v1/openai/gpt-4o:streamGenerateContent",
            headers={"x-gateway-api-key": "gw-secret"},
            json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
        )
    assert response.status_code == 404
