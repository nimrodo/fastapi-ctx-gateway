"""Runtime configuration for the gateway."""

from pathlib import Path
from typing import Literal

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

    # Every provider is an optional config group now (see ADR-0008): unset
    # (or empty-string) credentials mean that provider simply isn't
    # registered, never a boot failure. `create_app` refuses to boot only
    # if *no* provider ends up configured. Gemini used to be mandatory.
    gemini_upstream_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com"

    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Real OpenAI supports stream_options.include_usage; some self-hosted
    # "OpenAI-compatible" servers (vLLM, Ollama, ...) reject the field with
    # a 400. Set to false for those deployments — see ADR-0006.
    openai_include_usage: bool = True

    anthropic_api_key: SecretStr | None = None
    # None -> let the anthropic SDK use its own default base URL. Set it for
    # a proxy or an Anthropic-compatible gateway.
    anthropic_base_url: str | None = None
    # Anthropic's Messages API *requires* max_tokens; the neutral contract
    # makes it optional. This is the fallback used when a request omits
    # generation_config.max_output_tokens.
    anthropic_default_max_tokens: int = 4096

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
    # SecretStr-keyed (not dict[str, str]) so a repr()/log of Settings
    # masks tenant keys the same way the upstream provider keys already
    # are — see #24. SecretStr is hashable/equal-by-value, so it works as
    # a dict key with no other change to the lookup call sites.
    tenant_api_keys: dict[SecretStr, str]

    # None (the default) disables the semantic cache entirely — a missing
    # or unloadable model degrades to "always miss," never a boot failure,
    # matching the cache's fail-open contract. Point this at a real
    # exported model (see scripts/download_model.py) to enable it.
    embedding_model_path: Path | None = None

    # off: detector never runs (no measurable overhead). flag: detected,
    # logged, counted, request proceeds unmodified. block: additionally
    # short-circuits the request with a 400 (see guardrails.py and
    # routers/generate.py).
    prompt_injection_mode: Literal["off", "flag", "block"] = "off"

    # Unset (the default) means no ML backend is registered. Unlike the
    # semantic cache, this is a security control the deployer explicitly
    # opted into via prompt_injection_mode — silently downgrading to a
    # no-op would hide that; create_app() fails boot instead (see app.py)
    # if mode != "off" with no backend configured. "local_classifier" is
    # the only backend today; the Literal is shaped to grow (e.g. a future
    # LLM-self-check backend) without a schema break.
    prompt_injection_backend: Literal["local_classifier"] | None = None
    # Required when prompt_injection_backend="local_classifier". Same
    # local-ONNX-model pattern as embedding_model_path — onnxruntime is
    # already a core dependency, so no extra install is needed for this
    # backend specifically (see docs/tutorial/configuration.md and
    # scripts/download_model.py for how to export a real model).
    prompt_injection_model_path: Path | None = None
    # Probability (of the model's "injection" class) at or above which a
    # request is flagged. 0.5 is a neutral midpoint; tune against your own
    # exported model's calibration.
    prompt_injection_threshold: float = 0.5
    # Local model inference is CPU-bound and slower than the cache's
    # network-bound lookup, hence a separate, larger timeout rather than
    # reusing cache_lookup_timeout_ms.
    prompt_injection_timeout_ms: int = 200

    rpm_limit: int = 60
    tpm_limit: int = 100_000
    rate_limit_window_s: int = 60

    # Failed-auth backstop (auth.py::verify_api_key), independent of the
    # per-tenant-model limiter above: keyed per-source-IP, only consulted
    # (and only recorded against) when resolve_tenant fails — a request
    # with a valid key never touches this limiter — see docs/security.md.
    # Generous by default (20/window) so a legitimate tenant mistyping
    # their own key a few times, or several tenants behind one NAT'd IP,
    # isn't disproportionately locked out; tighten per-deployment if
    # brute-forcing is a real concern.
    auth_failure_rpm_limit: int = 20
    auth_failure_window_s: int = 60

    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_s: float = 30.0

    host: str = "0.0.0.0"  # noqa: S104 - intentional bind-all default for a gateway service
    port: int = 8000
    workers: int = 1
