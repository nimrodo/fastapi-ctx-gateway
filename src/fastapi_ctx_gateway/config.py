"""Runtime configuration for the gateway."""

from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "TokenBudgetConfig"]

_DEFAULT_MODEL_TOKEN_BUDGETS: dict[str, int] = {
    "gemini-3.7-flash": 32_000,
    "gemini-2.5-pro": 64_000,
}


class TokenBudgetConfig(BaseModel):
    """Per-model token caps that trigger context pruning.

    These are deliberately conservative relative to each model's real
    context window: they exist to trigger pruning early enough to help
    cache-key stability and latency, not to guard against hard overflow.
    """

    budgets: dict[str, int] = _DEFAULT_MODEL_TOKEN_BUDGETS
    default: int = 32_000

    def budget_for(self, model: str) -> int:
        """Return the configured pruning-trigger budget for a model name."""
        return self.budgets.get(model, self.default)


class Settings(BaseSettings):
    """Gateway configuration, loaded from the environment (prefix GATEWAY_)."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379"
    gemini_upstream_key: SecretStr
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    cache_distance_threshold: float = 0.10
    cache_ttl_s: int = 3600
    cache_temperature_threshold: float = 0.3
    cache_lookup_timeout_ms: int = 50

    token_budgets: TokenBudgetConfig = TokenBudgetConfig()

    # Maps gateway-issued API keys to tenant ids. Static config for now;
    # swap for a Redis-backed lookup once onboarding needs to be dynamic.
    # Required (no default): with no entries every request 401s (see
    # auth.py), so an empty mapping is never a working configuration —
    # better to fail at boot than ship a gateway nothing can call.
    tenant_api_keys: dict[str, str]

    # None (the default) disables the semantic cache entirely — a missing
    # or unloadable model degrades to "always miss," never a boot failure,
    # matching the cache's fail-open contract. Point this at a real
    # exported model (see scripts/download_model.py) to enable it.
    embedding_model_path: Path | None = None

    rpm_limit: int = 60
    tpm_limit: int = 100_000
    rate_limit_window_s: int = 60

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_s: float = 30.0

    host: str = "0.0.0.0"  # noqa: S104 - intentional bind-all default for a gateway service
    port: int = 8000
    workers: int = 1
