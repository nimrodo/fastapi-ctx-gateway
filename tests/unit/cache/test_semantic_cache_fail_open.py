"""Unit tests: SemanticCache.lookup/store fail open on any backend error or timeout.

Uses a fake redis_cache stub rather than a real RedisVL SemanticCache,
since RedisVL's SemanticCache connects to Redis eagerly at *construction*
time (even with overwrite=True) — there's no way to construct one against
an unreachable Redis to begin with. That eager-connect behavior is a real
startup-robustness concern handled separately in app.py's lifespan
(falls back to a disabled cache rather than failing to boot); these tests
target the call-level fail-open guarantee once a cache object exists.
"""

import asyncio

from fastapi_ctx_gateway.cache.semantic_cache import SemanticCache
from fastapi_ctx_gateway.schemas.neutral import TextPart, Turn


class _RaisingRedisCache:
    async def acheck(self, **kwargs) -> list:
        raise ConnectionError("simulated Redis outage")

    async def astore(self, **kwargs) -> str:
        raise ConnectionError("simulated Redis outage")


class _HangingRedisCache:
    async def acheck(self, **kwargs) -> list:
        await asyncio.sleep(10)
        return []

    async def astore(self, **kwargs) -> str:
        await asyncio.sleep(10)
        return ""


class _StubVectorizer:
    async def aembed(self, text: str, **kwargs) -> list[float]:
        return [0.1] * 384


def _turns() -> list[Turn]:
    return [Turn(role="user", parts=[TextPart(text="hello")])]


async def test_lookup_fails_open_on_backend_error() -> None:
    cache = SemanticCache(
        redis_cache=_RaisingRedisCache(),
        vectorizer=_StubVectorizer(),
        temperature_threshold=0.3,
        lookup_timeout_s=1.0,
    )
    hit = await cache.lookup(_turns(), "tenant-a", "model-x")
    assert hit is None


async def test_store_fails_open_on_backend_error() -> None:
    cache = SemanticCache(
        redis_cache=_RaisingRedisCache(),
        vectorizer=_StubVectorizer(),
        temperature_threshold=0.3,
        lookup_timeout_s=1.0,
    )
    await cache.store(_turns(), "tenant-a", "model-x", "hi", None)  # must not raise


async def test_lookup_fails_open_on_timeout() -> None:
    cache = SemanticCache(
        redis_cache=_HangingRedisCache(),
        vectorizer=_StubVectorizer(),
        temperature_threshold=0.3,
        lookup_timeout_s=0.05,
    )
    hit = await cache.lookup(_turns(), "tenant-a", "model-x")
    assert hit is None


async def test_on_fail_open_callback_fires_on_lookup_failure() -> None:
    calls = []
    cache = SemanticCache(
        redis_cache=_RaisingRedisCache(),
        vectorizer=_StubVectorizer(),
        temperature_threshold=0.3,
        lookup_timeout_s=1.0,
        on_fail_open=lambda: calls.append(1),
    )
    await cache.lookup(_turns(), "tenant-a", "model-x")
    assert len(calls) == 1


async def test_on_fail_open_callback_fires_on_store_failure() -> None:
    calls = []
    cache = SemanticCache(
        redis_cache=_RaisingRedisCache(),
        vectorizer=_StubVectorizer(),
        temperature_threshold=0.3,
        lookup_timeout_s=1.0,
        on_fail_open=lambda: calls.append(1),
    )
    await cache.store(_turns(), "tenant-a", "model-x", "hi", None)
    assert len(calls) == 1


async def test_on_fail_open_callback_not_called_on_genuine_miss() -> None:
    calls = []

    class _EmptyRedisCache:
        async def acheck(self, **kwargs) -> list:
            return []

    cache = SemanticCache(
        redis_cache=_EmptyRedisCache(),
        vectorizer=_StubVectorizer(),
        temperature_threshold=0.3,
        lookup_timeout_s=1.0,
        on_fail_open=lambda: calls.append(1),
    )
    hit = await cache.lookup(_turns(), "tenant-a", "model-x")
    assert hit is None
    assert len(calls) == 0
