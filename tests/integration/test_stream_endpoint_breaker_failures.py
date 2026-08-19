"""Router-level test: a hard Gemini failure (not just a truncated stream) trips the breaker."""

import httpx
import respx
from fastapi.testclient import TestClient
from support.neutral_sse import gemini_sse_event

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.circuit_breaker import CircuitState
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.deps import get_circuit_breaker


class _RecordingBreaker:
    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0

    @property
    def state(self) -> CircuitState:
        return CircuitState.CLOSED

    def allow_request(self) -> bool:
        return True

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self) -> None:
        self.failures += 1


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    return Settings()


def test_hard_upstream_failure_records_a_breaker_failure(monkeypatch) -> None:
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )

        app = create_app(_settings(monkeypatch))
        breaker = _RecordingBreaker()
        app.dependency_overrides[get_circuit_breaker] = lambda: breaker
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
            )

    assert breaker.failures == 1
    assert breaker.successes == 0


def test_clean_stream_records_a_breaker_success(monkeypatch) -> None:
    sse_body = gemini_sse_event(text="hi", finish_reason="STOP", total_tokens=3)
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        breaker = _RecordingBreaker()
        app.dependency_overrides[get_circuit_breaker] = lambda: breaker
        with TestClient(app) as client:
            client.post(
                "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
            )

    assert breaker.successes == 1
    assert breaker.failures == 0
