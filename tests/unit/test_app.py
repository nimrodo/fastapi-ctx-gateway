"""Tests for the create_app() factory."""

from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    return Settings()


def test_create_app_returns_bootable_app(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_app_builds_independent_instances(monkeypatch) -> None:
    """Two calls must not share mutable app state (no module-level singleton)."""
    app_a = create_app(_settings(monkeypatch))
    app_b = create_app(_settings(monkeypatch))
    assert app_a is not app_b


def test_openai_provider_not_registered_when_key_unset(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert set(app.state.providers) == {"gemini"}


def test_openai_provider_registered_when_key_set(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_OPENAI_API_KEY", "sk-test")
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert set(app.state.providers) == {"gemini", "openai"}


def test_openai_provider_not_registered_when_key_is_empty_string(monkeypatch) -> None:
    """An empty string is a likely misconfiguration, not an intentional key —
    treat it the same as unset rather than registering a provider that can
    never authenticate.
    """
    monkeypatch.setenv("GATEWAY_OPENAI_API_KEY", "")
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert set(app.state.providers) == {"gemini"}


def test_circuit_breakers_are_scoped_per_registered_provider(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_OPENAI_API_KEY", "sk-test")
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert set(app.state.circuit_breakers) == {"gemini", "openai"}
        assert app.state.circuit_breakers["gemini"] is not app.state.circuit_breakers["openai"]


def test_no_openai_circuit_breaker_when_openai_not_registered(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert set(app.state.circuit_breakers) == {"gemini"}
