"""Tests for AgentProvider: wraps a duck-typed LangChain-shaped object as a Provider."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi_ctx_gateway.providers.agent import (
    AgentProvider,
    _to_agent_messages,
    intermediate_step,
)
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


def test_to_agent_messages_emits_inline_binary_as_base64_content_block() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="Zm9v")])]
    )
    assert _to_agent_messages(request) == [
        (
            "user",
            [{"type": "image", "source_type": "base64", "data": "Zm9v", "mime_type": "image/png"}],
        )
    ]


def test_to_agent_messages_emits_by_reference_binary_as_url_content_block() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(
                role="user",
                parts=[BinaryPart(mime_type="image/png", uri="https://example.com/img.png")],
            )
        ]
    )
    assert _to_agent_messages(request) == [
        (
            "user",
            [
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "https://example.com/img.png",
                    "mime_type": "image/png",
                }
            ],
        )
    ]


def test_to_agent_messages_prefers_inline_data_when_both_data_and_uri_present() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(
                role="user",
                parts=[
                    BinaryPart(
                        mime_type="image/png", data="Zm9v", uri="https://example.com/img.png"
                    )
                ],
            )
        ]
    )
    block = _to_agent_messages(request)[0][1][0]
    assert block["source_type"] == "base64"
    assert block["data"] == "Zm9v"


def test_to_agent_messages_interleaves_text_and_binary_blocks_in_part_order() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(
                role="user",
                parts=[
                    TextPart(text="look at this:"),
                    BinaryPart(mime_type="image/png", data="Zm9v"),
                    TextPart(text="what is it?"),
                ],
            )
        ]
    )
    assert _to_agent_messages(request) == [
        (
            "user",
            [
                {"type": "text", "text": "look at this:"},
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": "Zm9v",
                    "mime_type": "image/png",
                },
                {"type": "text", "text": "what is it?"},
            ],
        )
    ]


def test_to_agent_messages_maps_mime_type_to_block_kind() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(role="user", parts=[BinaryPart(mime_type="audio/wav", data="Zm9v")]),
        ]
    )
    assert _to_agent_messages(request)[0][1][0]["type"] == "audio"

    request = NeutralGenerateRequest(
        turns=[
            Turn(role="user", parts=[BinaryPart(mime_type="application/pdf", data="Zm9v")]),
        ]
    )
    assert _to_agent_messages(request)[0][1][0]["type"] == "file"


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


async def test_stream_passes_binary_part_through_to_the_wrapped_agent() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="Zm9v")])]
    )
    agent = _AStreamAgent()
    provider = AgentProvider(name="my-agent", agent=agent)
    chunks = [c async for c in provider.stream("default", request)]
    assert _payload(chunks[0])["delta"]["parts"][0]["text"] == "Hel"
    assert agent.received == [
        (
            "user",
            [{"type": "image", "source_type": "base64", "data": "Zm9v", "mime_type": "image/png"}],
        )
    ]


# --- intermediate_step: the public contract for a node's custom stream_writer payload ---


def test_intermediate_step_carries_label_and_data() -> None:
    step = intermediate_step(label="tool_call", data={"tool": "search"})
    assert step.label == "tool_call"
    assert step.data == {"tool": "search"}


# --- streaming: LangGraph CompiledStateGraph path (intermediate events) ---


def _fake_chat_graph(node_body):
    """A minimal compiled StateGraph with one node, `node_body(state) -> dict`."""
    from langgraph.graph import END, START, MessagesState, StateGraph

    builder = StateGraph(MessagesState)
    builder.add_node("n", node_body)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile()


async def test_stream_translates_langgraph_intermediate_step_before_text() -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langgraph.config import get_stream_writer

    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

    async def node(state):
        writer = get_stream_writer()
        writer(intermediate_step(label="tool_call", data={"tool": "search"}))
        chunks = [c async for c in model.astream(state["messages"])]
        return {"messages": chunks}

    provider = AgentProvider(name="my-agent", agent=_fake_chat_graph(node))
    payloads = [_payload(c) async for c in provider.stream("default", _request())]

    intermediate_payloads = [p for p in payloads if p.get("intermediate")]
    assert len(intermediate_payloads) == 1
    assert intermediate_payloads[0]["intermediate"] == {
        "label": "tool_call",
        "data": {"tool": "search"},
    }
    # the intermediate event precedes the text delta the node emitted it before
    kinds = ["intermediate" if p.get("intermediate") else list(p)[0] for p in payloads]
    assert kinds.index("intermediate") < kinds.index("delta")


async def test_stream_drops_malformed_langgraph_custom_payload(caplog) -> None:
    from langgraph.config import get_stream_writer

    async def node(state):
        writer = get_stream_writer()
        writer({"not": "an intermediate_step()"})
        return {"messages": [("assistant", "Hello")]}

    provider = AgentProvider(name="my-agent", agent=_fake_chat_graph(node))
    with caplog.at_level("WARNING"):
        payloads = [_payload(c) async for c in provider.stream("default", _request())]
    assert not any(p.get("intermediate") for p in payloads)
    assert "dropping" in caplog.text.lower()


async def test_stream_translates_langgraph_text_deltas() -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    model = GenericFakeChatModel(messages=iter([AIMessage(content="Hello")]))

    async def node(state):
        chunks = [c async for c in model.astream(state["messages"])]
        return {"messages": chunks}

    provider = AgentProvider(name="my-agent", agent=_fake_chat_graph(node))
    chunks = [c async for c in provider.stream("default", _request())]
    texts = [_payload(c)["delta"]["parts"][0]["text"] for c in chunks[:-1]]
    assert "".join(texts) == "Hello"
    assert _payload(chunks[-1])["finish_reason"] == "stop"


# --- defaults ---


def test_cache_enabled_defaults_to_false() -> None:
    assert AgentProvider(name="my-agent", agent=_NothingAgent()).cache_enabled is False


def test_cache_enabled_can_be_opted_into() -> None:
    provider = AgentProvider(name="my-agent", agent=_NothingAgent(), cache_enabled=True)
    assert provider.cache_enabled is True
