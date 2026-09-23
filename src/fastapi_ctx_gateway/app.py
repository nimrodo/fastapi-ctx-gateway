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
from fastapi_ctx_gateway.guardrails import (
    InjectionDetector,
    LocalClassifierInjectionDetector,
    placeholder_tokenize,
)
from fastapi_ctx_gateway.observability.metrics import Metrics, build_metrics
from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.providers.registry import SPECS
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


def _build_injection_detector(settings: Settings, metrics: Metrics) -> InjectionDetector | None:
    """Build the injection detector, or None if detection is off.

    Unlike `_build_semantic_cache`, misconfiguration here raises rather
    than silently disabling: the cache is a pure optimization, but a
    deployer who set prompt_injection_mode to "flag"/"block" explicitly
    asked for detection — quietly running with no detector would hide a
    security control that looks enabled but isn't.
    """
    if settings.prompt_injection_backend is None:
        if settings.prompt_injection_mode != "off":
            raise RuntimeError(
                f"prompt_injection_mode={settings.prompt_injection_mode!r} requires "
                "prompt_injection_backend to be configured (e.g. 'local_classifier')"
            )
        return None
    if settings.prompt_injection_model_path is None:
        raise RuntimeError(
            "prompt_injection_backend='local_classifier' requires prompt_injection_model_path"
        )
    if not settings.prompt_injection_model_path.exists():
        raise RuntimeError(
            f"prompt-injection model not found at {settings.prompt_injection_model_path}"
        )
    session = ort.InferenceSession(str(settings.prompt_injection_model_path))
    return LocalClassifierInjectionDetector(
        session=session,
        tokenize=placeholder_tokenize,
        threshold=settings.prompt_injection_threshold,
        timeout_s=settings.prompt_injection_timeout_ms / 1000,
        on_fail_open=metrics.prompt_injection_fail_open.inc,
    )


def _registered_provider_names(settings: Settings) -> list[str]:
    """The provider names this Settings will register.

    Driven entirely by `providers/registry.py` — the single source of truth
    for which providers exist. Known before any async resource (http_client,
    etc.) exists and without importing any provider adapter module, so circuit
    breakers can be built for exactly these providers up front.
    """
    return [spec.name for spec in SPECS if spec.configured(settings)]


def _build_providers(settings: Settings, http_client: httpx.AsyncClient) -> dict[str, Provider]:
    # Each spec.build() lazily imports its adapter and turns a missing
    # dependency into an actionable error (install fastapi-ctx-gateway[<extra>]).
    return {
        spec.name: spec.build(settings, http_client) for spec in SPECS if spec.configured(settings)
    }


def create_app(settings: Settings) -> FastAPI:
    """Build a configured FastAPI application instance.

    No module-level app instance exists anywhere in this package — every
    caller (the CLI, tests, a consumer embedding this as a sub-app) builds
    its own via this factory, so independently-configured instances never
    share mutable state.
    """
    # No provider is mandatory (see ADR-0008), but a gateway with zero
    # providers can serve no traffic — fail at boot rather than 404 every
    # request.
    provider_names = _registered_provider_names(settings)
    if not provider_names:
        raise RuntimeError(
            "no LLM provider configured — set a provider key and install its extra: "
            "fastapi-ctx-gateway[gemini] / [openai] / [anthropic] / [all]"
        )

    # Synchronous, in-memory singletons — no async resources involved, so
    # built up front rather than in lifespan. One breaker per registered
    # provider (not one shared globally): an outage on one upstream must
    # not short-circuit requests to an unrelated one.
    circuit_breakers = {
        name: CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            reset_timeout_s=settings.circuit_breaker_reset_timeout_s,
        )
        for name in provider_names
    }
    metrics = build_metrics()
    # Synchronous like the breakers above — ONNX session construction needs
    # no async resources. Raises at create_app() time (not lifespan) if
    # prompt_injection_mode is on with no usable backend (see the
    # docstring on _build_injection_detector for why this fails loud
    # rather than degrading like the semantic cache does).
    injection_detector = _build_injection_detector(settings, metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once per app instance and shared across all requests via
        # app.state — never per-request (connection setup cost alone would
        # blow the latency budget if paid on every call).
        redis_client = redis_asyncio.from_url(settings.redis_url, decode_responses=False)
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            app.state.http_client = http_client
            app.state.redis_client = redis_client
            app.state.providers = _build_providers(settings, http_client)
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
                # Providers that own their own resources (the anthropic SDK's
                # httpx2 pool) expose aclose(); the shared httpx.AsyncClient is
                # closed by its own `async with` above.
                for provider in app.state.providers.values():
                    aclose = getattr(provider, "aclose", None)
                    if aclose is not None:
                        await aclose()
                await redis_client.aclose()

    app = FastAPI(title="fastapi-ctx-gateway", lifespan=lifespan)
    app.state.settings = settings
    app.state.circuit_breakers = circuit_breakers
    app.state.metrics = metrics
    app.state.injection_detector = injection_detector

    app.include_router(generate_router)
    register_exception_handlers(app)

    # Shares `metrics.registry` so gateway-specific counters and generic
    # HTTP metrics are both served from the same /metrics endpoint.
    Instrumentator(registry=metrics.registry).instrument(app).expose(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
