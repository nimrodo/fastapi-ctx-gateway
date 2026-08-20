"""The streamGenerateContent endpoint.

Thin orchestrator: sequences auth, rate-limit, pruning, cache, and the
provider call. Business logic lives in the service modules it calls into,
not here.
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from fastapi_ctx_gateway.auth import Tenant, verify_api_key
from fastapi_ctx_gateway.cache import CacheHit, SemanticCache
from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker, CircuitOpenError
from fastapi_ctx_gateway.deps import (
    get_circuit_breaker,
    get_metrics,
    get_provider,
    get_pruner,
    get_rate_limiter,
    get_semantic_cache,
)
from fastapi_ctx_gateway.observability.metrics import Metrics
from fastapi_ctx_gateway.observability.tracing import hit_path_span, pre_proxy_span
from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.proxy.streaming import StreamAccumulator, stream_with_retry
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter, RateLimitExceeded, TokenEstimator
from fastapi_ctx_gateway.schemas.neutral import (
    Delta,
    FinishReason,
    NeutralGenerateRequest,
    NeutralStreamEvent,
    Part,
    TextPart,
    Turn,
)

router = APIRouter()
_token_estimator = TokenEstimator()


@router.post("/v1/{provider_name}/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    request: NeutralGenerateRequest,
    tenant: Tenant = Depends(verify_api_key),
    provider: Provider = Depends(get_provider),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    pruner: TokenBudgetPruner = Depends(get_pruner),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
    circuit_breaker: CircuitBreaker = Depends(get_circuit_breaker),
    metrics: Metrics = Depends(get_metrics),
) -> StreamingResponse:
    """Proxy a streaming generate request through to the named provider.

    Sequenced deliberately: auth -> rate-limit check -> breaker precheck
    -> prune -> cache-eligibility -> cache lookup -> (hit: return) /
    (miss: proxy). A rejected request never reaches pruning or cache
    work, let alone the provider — and the breaker check (in-memory, O(1))
    happens before any of that, since it's the cheapest possible reject.
    Admission control is checked against the client's original, unpruned
    token estimate — pruning happens only after a request has already
    been admitted, and the cache is looked up against the *pruned*
    turns (better hit rate, cheaper to embed).
    """
    with pre_proxy_span():
        rate_limit_key = f"{tenant.api_key}:{model}"
        estimated_tokens = _token_estimator.estimate(turns=request.turns, system=request.system)
        decision = await rate_limiter.check(rate_limit_key, estimated_tokens)
        if not decision.allowed:
            metrics.rate_limit_rejected.inc()
            raise RateLimitExceeded(decision)

        if not circuit_breaker.allow_request():
            metrics.circuit_breaker_open.labels(provider=provider.name).inc()
            raise CircuitOpenError

        prune_result = pruner.prune(turns=request.turns, system=request.system, model=model)
        if prune_result.pruned:
            request = request.model_copy(update={"turns": prune_result.turns})
            metrics.prune_triggered.inc()

    cache_eligible = semantic_cache is not None and semantic_cache.is_eligible(
        tools=request.tools, generation_config=request.generation_config
    )
    if cache_eligible:
        assert semantic_cache is not None  # narrowed by cache_eligible
        with hit_path_span():
            hit = await semantic_cache.lookup(request.turns, tenant.id, model)
        if hit is not None:
            metrics.cache_hit.inc()
            return StreamingResponse(
                _synthesize_hit_stream(hit),
                media_type="text/event-stream",
                headers={"X-Cache": "HIT"},
            )

    metrics.cache_miss.inc()
    accumulator = StreamAccumulator()
    body = _stream_and_finalize(
        open_stream=lambda: provider.stream(model, request),
        accumulator=accumulator,
        rate_limiter=rate_limiter,
        rate_limit_key=rate_limit_key,
        estimated_tokens=estimated_tokens,
        semantic_cache=semantic_cache if cache_eligible else None,
        tenant_id=tenant.id,
        model=model,
        pruned_turns=request.turns,
        circuit_breaker=circuit_breaker,
    )
    return StreamingResponse(body, media_type="text/event-stream", headers={"X-Cache": "MISS"})


async def _stream_and_finalize(
    open_stream,
    accumulator: StreamAccumulator,
    rate_limiter: RateLimiter,
    rate_limit_key: str,
    estimated_tokens: int,
    semantic_cache: SemanticCache | None,
    tenant_id: str,
    model: str,
    pruned_turns: list[Turn],
    circuit_breaker: CircuitBreaker,
):
    """Stream (with bounded pre-stream retry) to the client, then finalize.

    Finalization (breaker recording, reconciliation, caching) runs after
    the loop, not as a detached background task: by then every byte has
    already reached the client, so it can't delay what they see, and
    awaiting it directly avoids managing orphaned-task lifecycle.
    stream_with_retry never raises — it always yields a terminal SSE
    error event on failure — so "finished_cleanly" is the single signal
    for whether this was a genuine success.
    """
    async for chunk in stream_with_retry(open_stream, accumulator):
        yield chunk
    if not accumulator.finished_cleanly:
        circuit_breaker.record_failure()
        return
    circuit_breaker.record_success()
    if accumulator.usage is not None:
        actual_tokens = accumulator.usage.total_tokens
        if actual_tokens is not None:
            await rate_limiter.reconcile(rate_limit_key, estimated_tokens, actual_tokens)
    if semantic_cache is not None:
        await semantic_cache.store(
            pruned_turns, tenant_id, model, accumulator.text, accumulator.usage
        )


def _synthesize_hit_stream(hit: CacheHit) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        parts: list[Part] = [TextPart(text=hit.response_text)]
        event = NeutralStreamEvent(
            delta=Delta(role="assistant", parts=parts),
            finish_reason=FinishReason.STOP,
            usage=hit.usage,
        )
        yield f"data: {event.model_dump_json(exclude_none=True)}\n\n".encode()

    return gen()
