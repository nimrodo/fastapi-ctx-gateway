"""FastAPI Depends() providers reading lifespan-owned singletons off app.state.

Internal wiring glue, not part of the public library API.
"""

from fastapi import Request

from fastapi_ctx_gateway.cache import SemanticCache
from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker
from fastapi_ctx_gateway.observability.metrics import Metrics
from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter


def get_gemini_client(request: Request) -> GeminiClient:
    """Return the shared GeminiClient built once during app startup."""
    client: GeminiClient = request.app.state.gemini_client
    return client


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
