"""A thin async client for Gemini's (stream)GenerateContent endpoints."""

from collections.abc import AsyncIterator

import httpx

from fastapi_ctx_gateway.schemas.gemini import GenerateContentRequest

__all__ = ["GeminiClient"]


class GeminiClient:
    """Wraps a shared httpx.AsyncClient to call Gemini's REST API.

    The http_client is injected (built once in the app's lifespan) rather
    than owned here, so its connection pool is shared across requests.
    """

    def __init__(self, http_client: httpx.AsyncClient, api_key: str, base_url: str) -> None:
        """Wrap a shared client with the credentials/base URL for one deployment."""
        self._http_client = http_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def stream_generate(
        self, model: str, request: GenerateContentRequest
    ) -> AsyncIterator[bytes]:
        """Open a streamGenerateContent SSE call and yield raw response bytes."""
        url = f"{self._base_url}/v1beta/models/{model}:streamGenerateContent"
        headers = {"x-goog-api-key": self._api_key, "content-type": "application/json"}
        body = request.model_dump(by_alias=True, exclude_none=True)
        async with self._http_client.stream(
            "POST", url, params={"alt": "sse"}, headers=headers, json=body
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
