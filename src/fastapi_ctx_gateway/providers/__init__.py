"""Pluggable upstream LLM providers: translate the neutral contract to/from one API."""

from typing import TYPE_CHECKING

from fastapi_ctx_gateway.providers.base import Provider

if TYPE_CHECKING:
    from fastapi_ctx_gateway.providers.agent import AgentProvider
    from fastapi_ctx_gateway.providers.anthropic import AnthropicProvider
    from fastapi_ctx_gateway.providers.gemini import GeminiProvider
    from fastapi_ctx_gateway.providers.openai import OpenAIProvider

__all__ = ["AgentProvider", "AnthropicProvider", "GeminiProvider", "OpenAIProvider", "Provider"]

# One lazy path for all four, not just AnthropicProvider: only `anthropic`
# is a real optional extra today (gemini/openai need nothing beyond the
# core httpx dependency, and agent.py already guards its one optional
# import, langgraph, internally) — but importing every adapter the same
# way here means that stays true by construction, not by each adapter
# happening to have no unguarded optional import today. Same lazy-import
# reasoning as ProviderSpec.build() in registry.py; see ADR-0008.
_MODULE_BY_NAME = {
    "AgentProvider": "fastapi_ctx_gateway.providers.agent",
    "AnthropicProvider": "fastapi_ctx_gateway.providers.anthropic",
    "GeminiProvider": "fastapi_ctx_gateway.providers.gemini",
    "OpenAIProvider": "fastapi_ctx_gateway.providers.openai",
}


def __getattr__(name: str) -> object:
    module_path = _MODULE_BY_NAME.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
