"""An open circuit breaker on one provider must never short-circuit another."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from support.neutral_sse import gemini_sse_event

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

pytestmark = pytest.mark.integration


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    monkeypatch.setenv("GATEWAY_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "1")
    return Settings()


def _post(client: TestClient, provider: str, model: str) -> httpx.Response:
    return client.post(
        f"/v1/{provider}/{model}:streamGenerateContent",
        headers={"x-gateway-api-key": "gw-secret"},
        json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
    )


def test_gemini_outage_does_not_trip_the_openai_breaker(monkeypatch) -> None:
    openai_sse_body = (
        b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    with (
        respx.mock(base_url="https://generativelanguage.googleapis.com") as gemini_mock,
        respx.mock(base_url="https://api.openai.com") as openai_mock,
    ):
        gemini_mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )
        openai_route = openai_mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, content=openai_sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app, raise_server_exceptions=False) as client:
            # First call fails and trips the (threshold=1) Gemini breaker;
            # the second call should be rejected by the now-open breaker.
            _post(client, "gemini", "gemini-3.7-flash")
            gemini_after_trip = _post(client, "gemini", "gemini-3.7-flash")
            openai_response = _post(client, "openai", "gpt-4o")

    assert gemini_after_trip.status_code == 503
    assert openai_response.status_code == 200
    assert openai_route.call_count == 1


def test_openai_outage_does_not_trip_the_gemini_breaker(monkeypatch) -> None:
    gemini_sse_body = gemini_sse_event(text="hi", finish_reason="STOP", total_tokens=3)
    with (
        respx.mock(base_url="https://generativelanguage.googleapis.com") as gemini_mock,
        respx.mock(base_url="https://api.openai.com") as openai_mock,
    ):
        gemini_route = gemini_mock.post(
            "/v1beta/models/gemini-3.7-flash:streamGenerateContent"
        ).mock(
            return_value=httpx.Response(
                200, content=gemini_sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        openai_mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app, raise_server_exceptions=False) as client:
            # First call fails and trips the (threshold=1) OpenAI breaker;
            # the second call should be rejected by the now-open breaker.
            _post(client, "openai", "gpt-4o")
            openai_after_trip = _post(client, "openai", "gpt-4o")
            gemini_response = _post(client, "gemini", "gemini-3.7-flash")

    assert openai_after_trip.status_code == 503
    assert gemini_response.status_code == 200
    assert gemini_route.call_count == 1
