"""Tests for Settings and TokenBudgetConfig defaults/overrides."""

import pytest
from pydantic import ValidationError

from fastapi_ctx_gateway.config import Settings, TokenBudgetConfig


def test_settings_loads_with_defaults(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    settings = Settings()
    assert settings.redis_url == "redis://localhost:6379"
    assert settings.cache_distance_threshold == 0.10
    assert settings.cache_ttl_s == 3600
    assert settings.cache_temperature_threshold == 0.3
    assert settings.cache_lookup_timeout_ms == 50


def test_settings_env_override(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    monkeypatch.setenv("GATEWAY_REDIS_URL", "redis://example:1234")
    settings = Settings()
    assert settings.redis_url == "redis://example:1234"


def test_settings_requires_tenant_api_keys(monkeypatch) -> None:
    """No tenants configured means every request would 401 anyway (see auth.py) —
    boot should fail loudly instead of shipping a gateway nothing can call.
    """
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.delenv("GATEWAY_TENANT_API_KEYS", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_token_budget_config_known_model() -> None:
    config = TokenBudgetConfig()
    assert config.budget_for("gemini-2.5-flash") == 32_000
    assert config.budget_for("gemini-2.5-pro") == 64_000


def test_token_budget_config_unknown_model_falls_back_to_default() -> None:
    config = TokenBudgetConfig()
    assert config.budget_for("some-future-model") == config.default
