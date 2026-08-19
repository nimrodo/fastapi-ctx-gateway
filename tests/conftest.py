"""Shared fixtures: a real Redis connection for integration tests."""

import os

import pytest
import redis.asyncio as redis_asyncio


@pytest.fixture
async def redis_client():
    """A real async Redis client, flushed before each test.

    Connects via REDIS_URL (default localhost:6379, matching
    docker-compose.yml). Skips the test with a clear message if
    unreachable, rather than hard-failing, so `uv run pytest` still runs
    the unit suite cleanly without Docker.
    """
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = redis_asyncio.from_url(url, decode_responses=False)
    try:
        await client.ping()
    except (redis_asyncio.RedisError, ConnectionError) as exc:
        await client.aclose()
        pytest.skip(f"Redis not reachable at {url} ({exc}); run `docker compose up -d redis`")
    await client.flushdb()
    yield client
    await client.aclose()
