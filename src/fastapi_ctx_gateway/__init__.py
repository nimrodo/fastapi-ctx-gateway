"""Context-aware agentic API gateway with pluggable LLM providers.

Public API surface: build a configured app via ``create_app(settings)``.
The CLI entrypoint (``main``) is intentionally not re-exported here — see
``fastapi_ctx_gateway.cli``.
"""

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.cache import OnnxVectorizer, SemanticCache
from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.providers.agent import (
    AgentProvider,
    intermediate_step,
    register_agent_provider,
)
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter, TokenEstimator

__all__ = [
    "AgentProvider",
    "CircuitBreaker",
    "OnnxVectorizer",
    "RateLimiter",
    "SemanticCache",
    "Settings",
    "TokenBudgetPruner",
    "TokenEstimator",
    "create_app",
    "intermediate_step",
    "register_agent_provider",
]
