"""Tests for the create_app() factory."""

import builtins
import sys

import pytest
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


# --- anthropic provider registration (mirrors the openai trio) ---


def test_anthropic_provider_not_registered_when_key_unset(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert "anthropic" not in app.state.providers


def test_anthropic_provider_registered_when_key_set(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_ANTHROPIC_API_KEY", "sk-ant-test")
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert set(app.state.providers) == {"gemini", "anthropic"}
        assert set(app.state.circuit_breakers) == {"gemini", "anthropic"}


def test_anthropic_provider_not_registered_when_key_is_empty_string(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_ANTHROPIC_API_KEY", "")
    app = create_app(_settings(monkeypatch))
    with TestClient(app):
        assert "anthropic" not in app.state.providers


# --- no provider mandatory (ADR-0008) ---


def test_boot_fails_when_no_provider_configured(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    with pytest.raises(RuntimeError, match="no LLM provider configured"):
        create_app(Settings())


def test_gemini_not_registered_when_key_unset(monkeypatch) -> None:
    """Gemini is no longer mandatory — an OpenAI-only deployment is valid."""
    monkeypatch.setenv("GATEWAY_OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    app = create_app(Settings())
    with TestClient(app):
        assert set(app.state.providers) == {"openai"}


def test_boot_fails_with_actionable_error_when_extra_missing(monkeypatch) -> None:
    """Key set but the provider's package absent -> actionable error, not ImportError."""
    monkeypatch.setenv("GATEWAY_ANTHROPIC_API_KEY", "sk-ant-test")
    # Drop cached modules so the lazy adapter import genuinely re-runs and hits
    # the patched importer, the same way it would with the extra uninstalled.
    for name in list(sys.modules):
        if name == "anthropic" or name.startswith("anthropic."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "fastapi_ctx_gateway.providers.anthropic", raising=False)
    real_import = builtins.__import__

    def _fail_anthropic(name, *args, **kwargs):
        if name == "anthropic" or name.startswith("anthropic."):
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_anthropic)
    # The lazy adapter import happens when the lifespan builds providers, so
    # the error surfaces on app startup rather than at create_app() time.
    app = create_app(_settings(monkeypatch))
    with pytest.raises(RuntimeError, match=r"fastapi-ctx-gateway\[anthropic\]"):  # noqa: SIM117
        with TestClient(app):
            pass
