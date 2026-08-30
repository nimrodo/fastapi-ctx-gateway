"""The registry of known providers — the single source of truth for which exist.

Importable with *no* provider extra installed: nothing here imports a provider
adapter module at load time. Each `ProviderSpec.build` does the adapter import
lazily and turns a missing dependency into an actionable error rather than a
bare `ImportError`. See ADR-0008 for why every provider is an opt-in extra.
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

from pydantic import SecretStr

from fastapi_ctx_gateway.config import Settings

if TYPE_CHECKING:
    import httpx

    from fastapi_ctx_gateway.providers.base import Provider

__all__ = ["SPECS", "ProviderSpec"]


class MissingProviderExtraError(RuntimeError):
    """A provider is configured but the package supplying it isn't installed."""

    def __init__(self, name: str, extra: str) -> None:
        super().__init__(
            f"provider {name!r} is configured but its dependency is not installed — "
            f"install fastapi-ctx-gateway[{extra}]"
        )


@dataclass(frozen=True)
class ProviderSpec:
    """How to detect and construct one provider without importing it eagerly."""

    name: str
    extra: str
    module: str
    configured: Callable[[Settings], bool]
    _factory: Callable[[Settings, "httpx.AsyncClient"], "Provider"]

    def build(self, settings: Settings, http_client: "httpx.AsyncClient") -> "Provider":
        """Import the adapter module (mapping ImportError -> actionable error) and construct it."""
        try:
            import_module(self.module)
        except ImportError as exc:
            raise MissingProviderExtraError(self.name, self.extra) from exc
        return self._factory(settings, http_client)


def _key_is_set(key: SecretStr | None) -> bool:
    # An empty string is almost certainly a misconfiguration (a blank .env
    # line, a secret that resolved empty) rather than an intentional key —
    # treat it the same as unset.
    return bool(key and key.get_secret_value())


def _build_gemini(settings: Settings, http_client: "httpx.AsyncClient") -> "Provider":
    from fastapi_ctx_gateway.providers.gemini import GeminiProvider

    assert settings.gemini_upstream_key is not None  # narrowed by `configured`
    return GeminiProvider(
        http_client=http_client,
        api_key=settings.gemini_upstream_key.get_secret_value(),
        base_url=settings.gemini_base_url,
    )


def _build_openai(settings: Settings, http_client: "httpx.AsyncClient") -> "Provider":
    from fastapi_ctx_gateway.providers.openai import OpenAIProvider

    assert settings.openai_api_key is not None  # narrowed by `configured`
    return OpenAIProvider(
        http_client=http_client,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        include_usage=settings.openai_include_usage,
    )


def _build_anthropic(settings: Settings, http_client: "httpx.AsyncClient") -> "Provider":
    # No `import anthropic` here — that would dodge build()'s ImportError
    # guard. The adapter module owns the SDK import; its own httpx2 pool is
    # closed by AnthropicProvider.aclose() from app.py's lifespan.
    from fastapi_ctx_gateway.providers.anthropic import AnthropicProvider

    assert settings.anthropic_api_key is not None  # narrowed by `configured`
    return AnthropicProvider.from_settings(
        api_key=settings.anthropic_api_key.get_secret_value(),
        base_url=settings.anthropic_base_url,
        default_max_tokens=settings.anthropic_default_max_tokens,
    )


SPECS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        name="gemini",
        extra="gemini",
        module="fastapi_ctx_gateway.providers.gemini",
        configured=lambda s: _key_is_set(s.gemini_upstream_key),
        _factory=_build_gemini,
    ),
    ProviderSpec(
        name="openai",
        extra="openai",
        module="fastapi_ctx_gateway.providers.openai",
        configured=lambda s: _key_is_set(s.openai_api_key),
        _factory=_build_openai,
    ),
    ProviderSpec(
        name="anthropic",
        extra="anthropic",
        module="fastapi_ctx_gateway.providers.anthropic",
        configured=lambda s: _key_is_set(s.anthropic_api_key),
        _factory=_build_anthropic,
    ),
)
