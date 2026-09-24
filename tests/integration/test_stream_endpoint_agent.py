"""Router-level test: a registered agent provider works end to end through the same route shape."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.providers.agent import register_agent_provider

pytestmark = pytest.mark.integration

FIXTURE_MODEL_PATH = str(
    Path(__file__).parent.parent / "fixtures" / "tiny_onnx_model" / "model.onnx"
)


class _StubAgent:
    """A minimal LangChain-Runnable-shaped stub: only .astream, content chunks."""

    class _Chunk:
        def __init__(self, content: str) -> None:
            self.content = content

    def __init__(self) -> None:
        self.calls = 0

    async def astream(self, messages: list[tuple[str, str]]) -> AsyncIterator[Any]:
        self.calls += 1
        yield self._Chunk("Hel")
        yield self._Chunk("lo")


def _settings(monkeypatch, *, with_cache: bool = False) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    if with_cache:
        monkeypatch.setenv("GATEWAY_EMBEDDING_MODEL_PATH", FIXTURE_MODEL_PATH)
        monkeypatch.setenv("GATEWAY_CACHE_DISTANCE_THRESHOLD", "0.05")
    return Settings()


def _post(client: TestClient, text: str = "hi", temperature: float = 0.0) -> Any:
    return client.post(
        "/v1/my-agent/default:streamGenerateContent",
        headers={"x-gateway-api-key": "gw-secret"},
        json={
            "turns": [{"role": "user", "parts": [{"type": "text", "text": text}]}],
            "generation_config": {"temperature": temperature},
        },
    )


def test_agent_route_streams_translated_neutral_body(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    agent = _StubAgent()
    register_agent_provider(app, name="my-agent", agent=agent)
    with TestClient(app) as client:
        response = _post(client)

    assert response.status_code == 200
    assert b'"text":"Hel"' in response.content
    assert b'"text":"lo"' in response.content
    assert b'"finish_reason":"stop"' in response.content
    assert agent.calls == 1


def test_agent_route_404s_when_not_registered(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        response = _post(client)
    assert response.status_code == 404


def test_agent_provider_bypasses_semantic_cache_by_default(monkeypatch) -> None:
    """cache_enabled defaults False for agent providers: a second identical
    request must still reach the agent (X-Cache: MISS both times), unlike
    the vendor providers' opt-in-by-default caching (test_stream_endpoint_cache.py).
    """
    app = create_app(_settings(monkeypatch, with_cache=True))
    agent = _StubAgent()
    register_agent_provider(app, name="my-agent", agent=agent)
    with TestClient(app) as client:
        first = _post(client)
        second = _post(client)

    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "MISS"
    assert agent.calls == 2


def test_agent_provider_can_opt_into_semantic_cache(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch, with_cache=True))
    agent = _StubAgent()
    register_agent_provider(app, name="my-agent", agent=agent, cache_enabled=True)
    with TestClient(app) as client:
        first = _post(client)
        second = _post(client)

    assert first.headers["x-cache"] == "MISS"
    assert second.headers["x-cache"] == "HIT"
    assert agent.calls == 1
