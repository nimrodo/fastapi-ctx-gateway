"""The streamGenerateContent endpoint.

Thin orchestrator: sequences auth, rate-limit, pruning, cache, and the
proxy call. Business logic lives in the service modules it calls into,
not here.
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from fastapi_ctx_gateway.auth import Tenant, verify_api_key
from fastapi_ctx_gateway.cache import CacheHit, SemanticCache
from fastapi_ctx_gateway.deps import (
    get_gemini_client,
    get_pruner,
    get_rate_limiter,
    get_semantic_cache,
)
from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.proxy.streaming import StreamAccumulator, tee_stream
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter, RateLimitExceeded, TokenEstimator
from fastapi_ctx_gateway.schemas.gemini import Content, GenerateContentRequest, Part

router = APIRouter()
_token_estimator = TokenEstimator()


@router.post("/v1/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    request: GenerateContentRequest,
    tenant: Tenant = Depends(verify_api_key),
    gemini_client: GeminiClient = Depends(get_gemini_client),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    pruner: TokenBudgetPruner = Depends(get_pruner),
    semantic_cache: SemanticCache | None = Depends(get_semantic_cache),
) -> StreamingResponse:
    """Proxy a streaming generate request through to Gemini.

    Sequenced deliberately: auth -> rate-limit check -> prune ->
    cache-eligibility -> cache lookup -> (hit: return) / (miss: proxy). A
    rejected request never reaches pruning or cache work, let alone
    Gemini. Admission control is checked against the client's original,
    unpruned token estimate — pruning happens only after a request has
    already been admitted, and the cache is looked up against the
    *pruned* contents (better hit rate, cheaper to embed).
    """
    rate_limit_key = f"{tenant.api_key}:{model}"
    estimated_tokens = _token_estimator.estimate(
        contents=request.contents, system_instruction=request.system_instruction
    )
    decision = await rate_limiter.check(rate_limit_key, estimated_tokens)
    if not decision.allowed:
        raise RateLimitExceeded(decision)

    prune_result = pruner.prune(
        contents=request.contents, system_instruction=request.system_instruction, model=model
    )
    if prune_result.pruned:
        request = request.model_copy(update={"contents": prune_result.contents})

    cache_eligible = semantic_cache is not None and semantic_cache.is_eligible(
        tools=request.tools, generation_config=request.generation_config
    )
    if cache_eligible:
        assert semantic_cache is not None  # narrowed by cache_eligible
        hit = await semantic_cache.lookup(request.contents, tenant.id, model)
        if hit is not None:
            return StreamingResponse(
                _synthesize_hit_stream(hit),
                media_type="text/event-stream",
                headers={"X-Cache": "HIT"},
            )

    accumulator = StreamAccumulator()
    upstream = gemini_client.stream_generate(model, request)
    body = _stream_and_finalize(
        upstream=upstream,
        accumulator=accumulator,
        rate_limiter=rate_limiter,
        rate_limit_key=rate_limit_key,
        estimated_tokens=estimated_tokens,
        semantic_cache=semantic_cache if cache_eligible else None,
        tenant_id=tenant.id,
        model=model,
        pruned_contents=request.contents,
    )
    return StreamingResponse(body, media_type="text/event-stream", headers={"X-Cache": "MISS"})


async def _stream_and_finalize(
    upstream,
    accumulator: StreamAccumulator,
    rate_limiter: RateLimiter,
    rate_limit_key: str,
    estimated_tokens: int,
    semantic_cache: SemanticCache | None,
    tenant_id: str,
    model: str,
    pruned_contents: list[Content],
):
    """Tee the stream to the client, then reconcile usage and (maybe) cache the response.

    Both run after the loop, not as detached background tasks: by then
    every byte has already reached the client, so neither can delay what
    they see, and awaiting them directly avoids managing orphaned-task
    lifecycle. No finishReason observed (disconnect, upstream error) ->
    neither reconciliation nor caching happens; a partial generation is
    never treated as complete.
    """
    async for chunk in tee_stream(upstream, accumulator):
        yield chunk
    if not accumulator.finished_cleanly:
        return
    if accumulator.usage:
        actual_tokens = accumulator.usage.get("totalTokenCount")
        if actual_tokens is not None:
            await rate_limiter.reconcile(rate_limit_key, estimated_tokens, actual_tokens)
    if semantic_cache is not None:
        await semantic_cache.store(
            pruned_contents, tenant_id, model, accumulator.text, accumulator.usage
        )


def _synthesize_hit_stream(hit: CacheHit) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        payload: dict = {
            "candidates": [
                {
                    "content": Content(
                        role="model", parts=[Part(text=hit.response_text)]
                    ).model_dump(by_alias=True),
                    "finishReason": "STOP",
                }
            ]
        }
        if hit.usage:
            payload["usageMetadata"] = hit.usage
        yield f"data: {json.dumps(payload)}\n\n".encode()

    return gen()
