"""End-to-end test: POST to the streaming endpoint passes through Gemini's SSE body."""

import httpx
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

SSE_BODY = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"}}]}\n\n'
    b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}],"role":"model"},'
    b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":5}}\n\n'
)


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv(
        "GATEWAY_TENANT_API_KEYS",
        '{"gw-secret": "tenant-a"}',
    )
    return Settings()


def test_passthrough_streams_gemini_body_byte_for_byte(monkeypatch) -> None:
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert response.status_code == 200
    assert response.content == SSE_BODY


def test_missing_api_key_returns_401(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        response = client.post(
            "/v1/gemini-2.5-flash:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        )
    assert response.status_code == 401
