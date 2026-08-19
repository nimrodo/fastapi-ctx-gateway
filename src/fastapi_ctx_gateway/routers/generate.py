"""The streamGenerateContent endpoint.

Thin orchestrator: sequences auth, (later milestones: rate-limit, circuit
breaker, pruning, cache) and the proxy call. Business logic lives in the
service modules it calls into, not here.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from fastapi_ctx_gateway.auth import Tenant, verify_api_key
from fastapi_ctx_gateway.deps import get_gemini_client, get_pruner, get_rate_limiter
from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.proxy.streaming import StreamAccumulator, tee_stream
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.ratelimit import RateLimiter, RateLimitExceeded, TokenEstimator
from fastapi_ctx_gateway.schemas.gemini import GenerateContentRequest

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
) -> StreamingResponse:
    """Proxy a streaming generate request through to Gemini.

    Sequenced deliberately: auth -> rate-limit check -> prune -> proxy. A
    rejected request never reaches pruning or the (later milestone) cache
    lookup, let alone Gemini. Admission control is checked against the
    client's original, unpruned token estimate — pruning happens only
    after a request has already been admitted.
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

    accumulator = StreamAccumulator()
    upstream = gemini_client.stream_generate(model, request)
    body = _stream_and_reconcile(
        upstream, accumulator, rate_limiter, rate_limit_key, estimated_tokens
    )
    return StreamingResponse(body, media_type="text/event-stream")


async def _stream_and_reconcile(
    upstream, accumulator: StreamAccumulator, rate_limiter: RateLimiter, key: str, estimated: int
):
    """Tee the stream to the client, then reconcile real usage once it finishes cleanly.

    Runs after the loop, not as a detached background task: by then every
    byte has already reached the client, so this can't delay what they
    see, and awaiting it directly avoids managing orphaned-task lifecycle.
    No finishReason observed (disconnect, upstream error) -> no
    reconciliation; the pre-call estimate stands.
    """
    async for chunk in tee_stream(upstream, accumulator):
        yield chunk
    if accumulator.finished_cleanly and accumulator.usage:
        actual_tokens = accumulator.usage.get("totalTokenCount")
        if actual_tokens is not None:
            await rate_limiter.reconcile(key, estimated, actual_tokens)
