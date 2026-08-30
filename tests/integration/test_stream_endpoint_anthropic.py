"""Router-level test: the Anthropic provider works end to end through the same route shape.

The anthropic SDK uses httpx2, which respx doesn't patch, so the upstream seam
here is an injected stub client rather than a mocked HTTP call (see ADR-0007).
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.integration


class _FakeStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for event in self._events:
            yield event

    async def close(self):
        pass


class _FakeClient:
    def __init__(self, events):
        self._events = events
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        return _FakeStream(self._events)

    async def close(self):
        pass


def _events():
    return [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=2,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                )
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Hello"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=3, input_tokens=None),
        ),
    ]


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    monkeypatch.setenv("GATEWAY_ANTHROPIC_API_KEY", "sk-ant-test")
    return Settings()


def test_anthropic_route_streams_translated_neutral_body(monkeypatch) -> None:
    app = create_app(_settings(monkeypatch))
    with TestClient(app) as client:
        app.state.providers["anthropic"] = AnthropicProvider(
            client=_FakeClient(_events()),  # type: ignore[arg-type]
            default_max_tokens=4096,
        )
        response = client.post(
            "/v1/anthropic/claude-opus-5:streamGenerateContent",
            headers={"x-gateway-api-key": "gw-secret"},
            json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
        )

    assert response.status_code == 200
    assert response.headers["x-cache"] == "MISS"
    assert b'"text":"Hello"' in response.content
    assert b'"finish_reason":"stop"' in response.content
    assert b'"total_tokens":5' in response.content


def test_anthropic_route_404s_without_credentials(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    app = create_app(Settings())  # no GATEWAY_ANTHROPIC_API_KEY set
    with TestClient(app) as client:
        response = client.post(
            "/v1/anthropic/claude-opus-5:streamGenerateContent",
            headers={"x-gateway-api-key": "gw-secret"},
            json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
        )
    assert response.status_code == 404
