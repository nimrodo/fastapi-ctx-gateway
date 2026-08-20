"""Router-level test: an open circuit breaker short-circuits before cache/Gemini."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.circuit_breaker import CircuitState
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.deps import get_circuit_breaker

pytestmark = pytest.mark.integration


class _AlwaysOpenBreaker:
    state = CircuitState.OPEN

    def allow_request(self) -> bool:
        return False

    def record_success(self) -> None:
        raise AssertionError("record_success should never be called when short-circuited")

    def record_failure(self) -> None:
        raise AssertionError("record_failure should never be called when short-circuited")


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    return Settings()


def test_open_breaker_short_circuits_before_gemini(monkeypatch) -> None:
    with respx.mock(
        base_url="https://generativelanguage.googleapis.com", assert_all_called=False
    ) as mock:
        route = mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n")
        )

        app = create_app(_settings(monkeypatch))
        app.dependency_overrides[get_circuit_breaker] = lambda: _AlwaysOpenBreaker()
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
            )

    assert response.status_code == 503
    assert route.call_count == 0
