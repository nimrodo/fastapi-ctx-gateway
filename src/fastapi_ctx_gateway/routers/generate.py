"""The streamGenerateContent endpoint.

Thin orchestrator: sequences auth, (later milestones: rate-limit, circuit
breaker, pruning, cache) and the proxy call. Business logic lives in the
service modules it calls into, not here.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from fastapi_ctx_gateway.auth import Tenant, verify_api_key
from fastapi_ctx_gateway.deps import get_gemini_client
from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.proxy.streaming import StreamAccumulator, tee_stream
from fastapi_ctx_gateway.schemas.gemini import GenerateContentRequest

router = APIRouter()


@router.post("/v1/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    request: GenerateContentRequest,
    tenant: Tenant = Depends(verify_api_key),
    gemini_client: GeminiClient = Depends(get_gemini_client),
) -> StreamingResponse:
    """Proxy a streaming generate request through to Gemini."""
    del tenant  # not yet used at this milestone; auth already gated the request
    accumulator = StreamAccumulator()
    upstream = gemini_client.stream_generate(model, request)
    return StreamingResponse(tee_stream(upstream, accumulator), media_type="text/event-stream")
