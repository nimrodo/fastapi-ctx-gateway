"""The FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import onnxruntime as ort
import redis.asyncio as redis_asyncio
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from redisvl.extensions.cache.llm import SemanticCache as RedisVLSemanticCache

from fastapi_ctx_gateway.cache import OnnxVectorizer, SemanticCache
from fastapi_ctx_gateway.cache.vectorizer import simple_char_code_tokenize
from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.errors import register_exception_handlers
from fastapi_ctx_gateway.observability.metrics import Metrics, build_metrics
from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter
from fastapi_ctx_gateway.routers.generate import router as generate_router

__all__ = ["create_app"]

logger = logging.getLogger(__name__)


def _build_semantic_cache(settings: Settings, metrics: Metrics) -> SemanticCache | None:
    """Build the semantic cache, or None if it can't be enabled right now.

    Disabled (not a startup failure) when no model is configured, the
    model file is missing, or Redis itself is unreachable — RedisVL's
    SemanticCache connects eagerly at construction, so a Redis outage at
    boot would otherwise crash the whole app. The cache must never become
    a hard dependency for serving traffic, at startup or at request time.
    """
    if settings.embedding_model_path is None:
        return None
    if not settings.embedding_model_path.exists():
        logger.warning(
            "embedding model not found at %s; semantic cache disabled",
            settings.embedding_model_path,
        )
        return None
    try:
        session = ort.InferenceSession(str(settings.embedding_model_path))
        vectorizer = OnnxVectorizer(session=session, tokenize=simple_char_code_tokenize, dims=384)
        redis_cache = RedisVLSemanticCache(
            name="fastapi_ctx_gateway_semantic_cache",
            distance_threshold=settings.cache_distance_threshold,
            ttl=settings.cache_ttl_s,
            vectorizer=vectorizer,
            filterable_fields=[
                {"name": "tenant_id", "type": "tag"},
                {"name": "model", "type": "tag"},
            ],
            redis_url=settings.redis_url,
        )
        return SemanticCache(
            redis_cache=redis_cache,
            vectorizer=vectorizer,
            temperature_threshold=settings.cache_temperature_threshold,
            lookup_timeout_s=settings.cache_lookup_timeout_ms / 1000,
            on_fail_open=metrics.vector_store_fail_open.inc,
        )
    except Exception:
        logger.warning("failed to initialize semantic cache; disabling it", exc_info=True)
        return None


def create_app(settings: Settings) -> FastAPI:
    """Build a configured FastAPI application instance.

    No module-level app instance exists anywhere in this package — every
    caller (the CLI, tests, a consumer embedding this as a sub-app) builds
    its own via this factory, so independently-configured instances never
    share mutable state.
    """
    # Synchronous, in-memory singletons — no async resources involved, so
    # built up front rather than in lifespan.
    circuit_breaker = CircuitBreaker(
        failure_threshold=settings.circuit_breaker_failure_threshold,
        reset_timeout_s=settings.circuit_breaker_reset_timeout_s,
    )
    metrics = build_metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once per app instance and shared across all requests via
        # app.state — never per-request (connection setup cost alone would
        # blow the latency budget if paid on every call).
        redis_client = redis_asyncio.from_url(settings.redis_url, decode_responses=False)
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            app.state.http_client = http_client
            app.state.redis_client = redis_client
            app.state.gemini_client = GeminiClient(
                http_client=http_client,
                api_key=settings.gemini_upstream_key.get_secret_value(),
                base_url=settings.gemini_base_url,
            )
            app.state.rate_limiter = RateLimiter(
                redis_client=redis_client,
                rpm_limit=settings.rpm_limit,
                tpm_limit=settings.tpm_limit,
                window_s=settings.rate_limit_window_s,
            )
            app.state.pruner = TokenBudgetPruner(settings.token_budgets)
            app.state.semantic_cache = _build_semantic_cache(settings, metrics)
            try:
                yield
            finally:
                await redis_client.aclose()

    app = FastAPI(title="fastapi-ctx-gateway", lifespan=lifespan)
    app.state.settings = settings
    app.state.circuit_breaker = circuit_breaker
    app.state.metrics = metrics

    app.include_router(generate_router)
    register_exception_handlers(app)

    # Shares `metrics.registry` so gateway-specific counters and generic
    # HTTP metrics are both served from the same /metrics endpoint.
    Instrumentator(registry=metrics.registry).instrument(app).expose(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
