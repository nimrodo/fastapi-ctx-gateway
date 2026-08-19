"""FastAPI Depends() providers reading lifespan-owned singletons off app.state.

Internal wiring glue, not part of the public library API.
"""

from fastapi import Request

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
