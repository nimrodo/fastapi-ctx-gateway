"""Tests for AgentProvider: wraps a duck-typed LangChain-shaped object as a Provider."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi_ctx_gateway.providers.agent import AgentProvider, _to_agent_messages
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    NeutralGenerateRequest,
    TextPart,
    Turn,
)


def _payload(chunk: bytes) -> dict[str, Any]:
    return json.loads(chunk.removeprefix(b"data: ").strip())


def _request(
    *, system: list[TextPart] | None = None, user_text: str = "hi"
) -> NeutralGenerateRequest:
    return NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text=user_text)])],
        system=system,
    )


# --- request translation: neutral -> (role, content) message tuples ---


def test_to_agent_messages_maps_turns_to_role_content_tuples() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(role="user", parts=[TextPart(text="hi")]),
            Turn(role="assistant", parts=[TextPart(text="hello")]),
        ]
    )
    assert _to_agent_messages(request) == [("user", "hi"), ("assistant", "hello")]


def test_to_agent_messages_prepends_system_as_a_system_message() -> None:
    request = _request(system=[TextPart(text="be terse")])
    assert _to_agent_messages(request)[0] == ("system", "be terse")


def test_to_agent_messages_joins_multiple_text_parts_in_one_turn() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="a"), TextPart(text="b")])]
    )
    assert _to_agent_messages(request) == [("user", "ab")]


def test_to_agent_messages_rejects_binary_parts() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="Zm9v")])]
    )
    try:
        _to_agent_messages(request)
    except ValueError as exc:
        assert "text-only" in str(exc)
    else:
        raise AssertionError("expected ValueError for a binary part")


# --- streaming: duck-typed .astream / .stream / .ainvoke / .invoke fallback chain ---


class _AStreamAgent:
    """Shaped like a LangChain Runnable exposing only .astream (chunks with .content)."""

    class _Chunk:
        def __init__(self, content: str) -> None:
            self.content = content

    async def astream(self, messages: list[tuple[str, str]]) -> AsyncIterator[Any]:
        self.received = messages
        yield self._Chunk("Hel")
        yield self._Chunk("lo")


class _StreamAgent:
    """Shaped like a plain sync Runnable exposing only .stream (str chunks)."""

    def stream(self, messages: list[tuple[str, str]]):
        self.received = messages
        yield "Hel"
        yield "lo"


class _AinvokeAgent:
    """Exposes only .ainvoke, returning a dict-shaped result."""

    async def ainvoke(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        self.received = messages
        return {"content": "Hello"}


class _InvokeAgent:
    """Exposes only sync .invoke."""

    def invoke(self, messages: list[tuple[str, str]]) -> str:
        self.received = messages
        return "Hello"


class _RaisingAgent:
    async def astream(self, messages: list[tuple[str, str]]) -> AsyncIterator[Any]:
        yield "par"
        raise RuntimeError("boom")


class _NothingAgent:
    """Has none of the recognized methods."""


async def test_stream_prefers_astream_and_translates_content_chunks() -> None:
    agent = _AStreamAgent()
    provider = AgentProvider(name="my-agent", agent=agent)
    chunks = [c async for c in provider.stream("default", _request())]
    texts = [_payload(c)["delta"]["parts"][0]["text"] for c in chunks[:-1]]
    assert texts == ["Hel", "lo"]
    assert _payload(chunks[-1])["finish_reason"] == "stop"
    assert agent.received == [("user", "hi")]


async def test_stream_falls_back_to_sync_stream() -> None:
    agent = _StreamAgent()
    provider = AgentProvider(name="my-agent", agent=agent)
    chunks = [c async for c in provider.stream("default", _request())]
    texts = [_payload(c)["delta"]["parts"][0]["text"] for c in chunks[:-1]]
    assert texts == ["Hel", "lo"]


async def test_stream_falls_back_to_ainvoke() -> None:
    agent = _AinvokeAgent()
    provider = AgentProvider(name="my-agent", agent=agent)
    chunks = [c async for c in provider.stream("default", _request())]
    assert _payload(chunks[0])["delta"]["parts"][0]["text"] == "Hello"
    assert _payload(chunks[-1])["finish_reason"] == "stop"


async def test_stream_falls_back_to_sync_invoke() -> None:
    agent = _InvokeAgent()
    provider = AgentProvider(name="my-agent", agent=agent)
    chunks = [c async for c in provider.stream("default", _request())]
    assert _payload(chunks[0])["delta"]["parts"][0]["text"] == "Hello"


async def test_stream_never_raises_translates_agent_exception_to_neutral_error() -> None:
    provider = AgentProvider(name="my-agent", agent=_RaisingAgent())
    chunks = [c async for c in provider.stream("default", _request())]
    last = _payload(chunks[-1])
    assert last["error"]["message"] == "boom"
    assert last["error"]["type"] == "agent_error"


async def test_stream_never_raises_on_unsupported_agent_shape() -> None:
    provider = AgentProvider(name="my-agent", agent=_NothingAgent())
    chunks = [c async for c in provider.stream("default", _request())]
    assert len(chunks) == 1
    assert "error" in _payload(chunks[0])


async def test_stream_never_raises_on_binary_part_rejects_with_neutral_error() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="Zm9v")])]
    )
    provider = AgentProvider(name="my-agent", agent=_AStreamAgent())
    chunks = [c async for c in provider.stream("default", request)]
    assert len(chunks) == 1
    payload = _payload(chunks[0])
    assert payload["error"]["type"] == "agent_input_unsupported"


# --- defaults ---


def test_cache_enabled_defaults_to_false() -> None:
    assert AgentProvider(name="my-agent", agent=_NothingAgent()).cache_enabled is False


def test_cache_enabled_can_be_opted_into() -> None:
    provider = AgentProvider(name="my-agent", agent=_NothingAgent(), cache_enabled=True)
    assert provider.cache_enabled is True
