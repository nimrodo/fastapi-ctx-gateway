"""Router-level test: domain counters increment on the right paths, and /metrics serves them."""

import httpx
import respx
from fastapi.testclient import TestClient
from support.neutral_sse import gemini_sse_event

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.circuit_breaker import CircuitState
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.deps import get_circuit_breaker


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    return Settings()


def _counter_value(counter) -> float:
    return counter.collect()[0].samples[0].value


def _labeled_counter_value(counter, **labels) -> float:
    for sample in counter.collect()[0].samples:
        if sample.labels == labels:
            return sample.value
    return 0.0


class _AlwaysOpenBreaker:
    state = CircuitState.OPEN

    def allow_request(self) -> bool:
        return False

    def record_success(self) -> None:
        raise AssertionError("record_success should never be called when short-circuited")

    def record_failure(self) -> None:
        raise AssertionError("record_failure should never be called when short-circuited")


def test_miss_path_increments_cache_miss_counter(monkeypatch) -> None:
    sse_body = gemini_sse_event(text="hi", finish_reason="STOP", total_tokens=3)
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )
        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            client.post(
                "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
            )
        assert _counter_value(app.state.metrics.cache_miss) == 1
        assert _counter_value(app.state.metrics.cache_hit) == 0


def test_circuit_breaker_open_counter_is_labeled_by_provider(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    app.dependency_overrides[get_circuit_breaker] = lambda: _AlwaysOpenBreaker()
    with TestClient(app) as client:
        client.post(
            "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
            headers={"x-gateway-api-key": "gw-secret"},
            json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
        )
    assert _labeled_counter_value(app.state.metrics.circuit_breaker_open, provider="gemini") == 1


def test_metrics_endpoint_serves_domain_counters(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "cache_hit_total" in body
    assert "cache_miss_total" in body
    assert "prune_triggered_total" in body
    assert "rate_limit_rejected_total" in body
    assert "circuit_breaker_open_total" in body
    assert "vector_store_fail_open_total" in body
