"""Eligibility gating + a fail-open wrapper around RedisVL's AsyncSemanticCache."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from redisvl.extensions.cache.llm import SemanticCache as RedisVLSemanticCache
from redisvl.query.filter import Tag

from fastapi_ctx_gateway.cache.serialize import canonicalize_contents
from fastapi_ctx_gateway.cache.vectorizer import OnnxVectorizer
from fastapi_ctx_gateway.schemas.gemini import Content, GenerationConfig

__all__ = ["CacheHit", "SemanticCache", "is_cache_eligible"]

logger = logging.getLogger(__name__)


def is_cache_eligible(
    tools: list[dict[str, Any]] | None,
    generation_config: GenerationConfig | None,
    temperature_threshold: float,
) -> bool:
    """A request is cache-eligible only when its output is meant to be near-deterministic.

    Bypasses (returns False) when tools are enabled (a cached response
    can't reflect a different tool availability) or temperature is unset
    or above the threshold — an unset temperature is treated as "not
    explicitly low," not as an invitation to cache.
    """
    if tools:
        return False
    temperature = generation_config.temperature if generation_config else None
    if temperature is None:
        return False
    return temperature <= temperature_threshold


@dataclass(frozen=True)
class CacheHit:
    """A cached response, ready to be replayed to the client."""

    response_text: str
    usage: dict[str, Any] | None


class SemanticCache:
    """Fail-open: any lookup/store failure (timeout, connection error, ...) is swallowed.

    The cache is a pure optimization; it must never become a hard
    dependency for serving traffic.
    """

    def __init__(
        self,
        redis_cache: RedisVLSemanticCache,
        vectorizer: OnnxVectorizer,
        temperature_threshold: float,
        lookup_timeout_s: float,
    ) -> None:
        """Wire in the RedisVL cache, the vectorizer, and the fail-open bound."""
        self._redis_cache = redis_cache
        self._vectorizer = vectorizer
        self._temperature_threshold = temperature_threshold
        self._lookup_timeout_s = lookup_timeout_s

    def is_eligible(
        self, tools: list[dict[str, Any]] | None, generation_config: GenerationConfig | None
    ) -> bool:
        """See is_cache_eligible."""
        return is_cache_eligible(tools, generation_config, self._temperature_threshold)

    async def lookup(self, contents: list[Content], tenant_id: str, model: str) -> CacheHit | None:
        """Look up a semantically similar cached response, or None on miss/failure/timeout."""
        try:
            return await asyncio.wait_for(
                self._lookup(contents, tenant_id, model), timeout=self._lookup_timeout_s
            )
        except Exception:
            logger.warning("semantic cache lookup failed; failing open", exc_info=True)
            return None

    async def _lookup(self, contents: list[Content], tenant_id: str, model: str) -> CacheHit | None:
        text = canonicalize_contents(contents)
        vector = await self._vectorizer.aembed(text)
        filter_expression = (Tag("tenant_id") == tenant_id) & (Tag("model") == model)
        results = await self._redis_cache.acheck(vector=vector, filter_expression=filter_expression)
        if not results:
            return None
        result = results[0]
        return CacheHit(response_text=result["response"], usage=result.get("metadata"))

    async def store(
        self,
        contents: list[Content],
        tenant_id: str,
        model: str,
        response_text: str,
        usage: dict[str, Any] | None,
    ) -> None:
        """Store a completed response for future lookups. Failure is logged, never raised."""
        try:
            await self._store(contents, tenant_id, model, response_text, usage)
        except Exception:
            logger.warning("semantic cache store failed; skipping", exc_info=True)

    async def _store(
        self,
        contents: list[Content],
        tenant_id: str,
        model: str,
        response_text: str,
        usage: dict[str, Any] | None,
    ) -> None:
        text = canonicalize_contents(contents)
        vector = await self._vectorizer.aembed(text)
        await self._redis_cache.astore(
            prompt=text,
            response=response_text,
            vector=vector,
            filters={"tenant_id": tenant_id, "model": model},
            metadata=usage,
        )
