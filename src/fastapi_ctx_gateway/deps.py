"""FastAPI Depends() providers reading lifespan-owned singletons off app.state.

Internal wiring glue, not part of the public library API.
"""

from fastapi import Request

from fastapi_ctx_gateway.cache import SemanticCache
from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker
from fastapi_ctx_gateway.errors import ProviderNotFoundError
from fastapi_ctx_gateway.observability.metrics import Metrics
from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter


def get_provider(provider_name: str, request: Request) -> Provider:
    """Look up the requested provider by its path segment, or 404 if unknown."""
    providers: dict[str, Provider] = request.app.state.providers
    provider = providers.get(provider_name)
    if provider is None:
        raise ProviderNotFoundError(provider_name)
    return provider


def get_rate_limiter(request: Request) -> RateLimiter:
    """Return the shared RateLimiter built once during app startup."""
    limiter: RateLimiter = request.app.state.rate_limiter
    return limiter


def get_pruner(request: Request) -> TokenBudgetPruner:
    """Return the shared TokenBudgetPruner built once during app startup."""
    pruner: TokenBudgetPruner = request.app.state.pruner
    return pruner


def get_semantic_cache(request: Request) -> SemanticCache | None:
    """Return the shared SemanticCache, or None if it's disabled for this deployment."""
    cache: SemanticCache | None = request.app.state.semantic_cache
    return cache


def get_circuit_breaker(request: Request) -> CircuitBreaker:
    """Return the shared, per-worker CircuitBreaker built once during app startup."""
    breaker: CircuitBreaker = request.app.state.circuit_breaker
    return breaker


def get_metrics(request: Request) -> Metrics:
    """Return the shared Prometheus counters built once during app startup."""
    metrics: Metrics = request.app.state.metrics
    return metrics
