"""End-to-end test: POST to the streaming endpoint translates Gemini's SSE to neutral SSE."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from support.neutral_sse import gemini_sse_event, neutral_sse_event

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

pytestmark = pytest.mark.integration

GEMINI_SSE_BODY = gemini_sse_event(text="Hel") + gemini_sse_event(
    text="lo", finish_reason="STOP", total_tokens=5
)
EXPECTED_NEUTRAL_BODY = neutral_sse_event(text="Hel") + neutral_sse_event(
    text="lo", finish_reason="stop", total_tokens=5
)


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv(
        "GATEWAY_TENANT_API_KEYS",
        '{"gw-secret": "tenant-a"}',
    )
    return Settings()


def test_passthrough_streams_translated_neutral_body(monkeypatch) -> None:
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=GEMINI_SSE_BODY, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
            )

    assert response.status_code == 200
    assert response.content == EXPECTED_NEUTRAL_BODY


def test_unknown_provider_returns_404(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        response = client.post(
            "/v1/openai/gpt-4o:streamGenerateContent",
            headers={"x-gateway-api-key": "gw-secret"},
            json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
        )
    assert response.status_code == 404


def test_missing_api_key_returns_401(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        response = client.post(
            "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
            json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
        )
    assert response.status_code == 401
