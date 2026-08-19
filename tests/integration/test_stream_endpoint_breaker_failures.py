"""Router-level test: a hard Gemini failure (not just a truncated stream) trips the breaker."""

import httpx
import respx
from fastapi.testclient import TestClient

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
        mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(500, content=b"internal error")
        )

        app = create_app(_settings(monkeypatch))
        breaker = _RecordingBreaker()
        app.dependency_overrides[get_circuit_breaker] = lambda: breaker
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert breaker.failures == 1
    assert breaker.successes == 0


def test_clean_stream_records_a_breaker_success(monkeypatch) -> None:
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},'
        b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":3}}\n\n'
    )
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        breaker = _RecordingBreaker()
        app.dependency_overrides[get_circuit_breaker] = lambda: breaker
        with TestClient(app) as client:
            client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert breaker.successes == 1
    assert breaker.failures == 0
