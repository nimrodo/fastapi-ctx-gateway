"""Shared fixtures: a real Redis connection for integration tests, and example-app loading."""

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import redis.asyncio as redis_asyncio


@pytest.fixture
def load_example_app(monkeypatch) -> Callable[[Path, str], Any]:
    """Factory fixture: import a standalone `examples/*/app.py` by file path and return its `app`.

    Shared by an example's unit smoke test (no Redis) and its integration
    test (real endpoint calls) — both need the same importlib-by-path
    loading, since examples aren't part of the installed package.
    """
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')

    def _load(app_path: Path, module_name: str) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, app_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.app

    return _load


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
