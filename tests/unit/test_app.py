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
