"""Agent provider adapter: wraps a developer's own in-process LangChain-shaped object.

Unlike `GeminiProvider`/`OpenAIProvider`/`AnthropicProvider`, this adapter has
no upstream HTTP call and no credential to configure — the "upstream" is a
live Python object (a LangChain `Runnable`, a LangGraph `CompiledGraph`, or
anything shaped like one) the developer already built and hands in directly.
It is registered via `register_agent_provider()` from the developer's own
FastAPI startup code, following the "mount as a library" convention
(docs/tutorial/using-as-a-library.md) rather than `providers/registry.py`'s
Settings-driven `ProviderSpec` table, since there is no credential to detect.

Text-only for v1 (multimodal is tracked separately, see issue #27) and
duck-typed: no hard import of `langchain_core`/`langgraph` types. Anything
exposing `.astream`/`.stream`/`.ainvoke`/`.invoke` works, including a plain
LangChain `Runnable`. LangGraph-specific intermediate-event streaming
(`stream_mode=["messages", "custom"]`) is a separate adapter path, not
implemented here (see the map, issue #28).
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.providers.sse import neutral_error_event
from fastapi_ctx_gateway.schemas.neutral import (
    Delta,
    FinishReason,
    IntermediateStep,
    NeutralGenerateRequest,
    NeutralStreamEvent,
    Part,
    TextPart,
)

__all__ = ["AgentProvider", "intermediate_step", "register_agent_provider"]

logger = logging.getLogger(__name__)

# `langgraph` stays a soft dependency: a plain-Runnable/duck-typed caller
# never needs it installed. `CompiledStateGraph` is `None` when absent, and
# `_is_langgraph_compiled_graph` treats that as "never a LangGraph object."
try:
    from langgraph.graph.state import CompiledStateGraph
except ImportError:  # pragma: no cover - exercised by not installing langgraph
    CompiledStateGraph = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class _IntermediateStepPayload:
    """The one shape `_stream_langgraph_events` recognizes on a "custom" chunk.

    Deliberately a frozen dataclass, not a bare dict: a developer's own
    `get_stream_writer()` payload could easily be a dict for unrelated
    reasons, and a dict-shape sniff (e.g. "has a 'label' key") would guess at
    intent instead of requiring an explicit, unambiguous opt-in.
    """

    label: str | None
    data: Any


def intermediate_step(label: str | None, data: Any) -> _IntermediateStepPayload:
    """Build the payload a LangGraph node passes to `get_stream_writer()`.

    Call this from inside a node to surface a custom step on the gateway's
    `intermediate` SSE field::

        writer = get_stream_writer()
        writer(intermediate_step(label="tool_call", data={"tool": "search"}))

    A `writer()` payload that isn't produced by this helper is dropped (with
    a logged warning), not guessed at — see `_stream_langgraph_events`.
    """
    return _IntermediateStepPayload(label=label, data=data)


class AgentProvider(Provider):
    """Adapts a duck-typed LangChain-shaped object to the `Provider` contract.

    `model` (the second path segment, `/v1/{name}/{model}:streamGenerateContent`)
    is accepted but ignored: one registration binds to one already-built
    agent object, not a family of selectable models.
    """

    # Opt-out by default, unlike the vendor providers (Provider.cache_enabled
    # defaults True there): an agent's own code can have arbitrary side
    # effects (tool calls), so assuming a cached response is safe to replay
    # for "the same prompt" isn't a safe default the way it is for a
    # stateless chat completion. Pass cache_enabled=True to opt back in.
    cache_enabled: bool = False

    # Deliberately no pre-stream retry (unlike the other three providers):
    # retrying arbitrary agent code risks re-triggering side effects a tool
    # call already caused on the failed attempt.

    def __init__(self, name: str, agent: Any, cache_enabled: bool = False) -> None:
        """Wrap `agent` (any object shaped like a LangChain `Runnable`)."""
        self.name = name
        self._agent = agent
        self.cache_enabled = cache_enabled

    async def stream(self, model: str, request: NeutralGenerateRequest) -> AsyncIterator[bytes]:
        """Invoke the wrapped agent and yield neutral SSE bytes. Never raises."""
        try:
            messages = _to_agent_messages(request)
        except ValueError as exc:
            yield neutral_error_event(str(exc), None, error_type="agent_input_unsupported")
            return
        try:
            if _is_langgraph_compiled_graph(self._agent):
                async for event in _stream_langgraph_events(self._agent, messages):
                    yield event
            else:
                async for text in _stream_text(self._agent, messages):
                    yield _text_event(text)
            yield _final_event()
        except Exception as exc:  # noqa: BLE001 - stream() must never raise, see Provider
            yield neutral_error_event(str(exc), None, error_type="agent_error")


def register_agent_provider(
    app: FastAPI,
    name: str,
    agent: Any,
    *,
    cache_enabled: bool = False,
    circuit_breaker_failure_threshold: int | None = None,
    circuit_breaker_reset_timeout_s: float | None = None,
) -> AgentProvider:
    """Register a developer's own agent as a gateway provider, in-place on `app`.

    Call any time after `create_app()` returns, before or after the app's
    lifespan has started — `app.state.providers` and `app.state.circuit_breakers`
    are both populated synchronously by `create_app()` itself precisely so
    this works without depending on lifespan ordering (an agent provider has
    no async resource, like an httpx client, to wait for). This mirrors
    `app.include_router()`/`app.mount()`: a call made against the app object
    directly, not a Settings/env-var entry in `providers/registry.py`'s
    `SPECS` table, since the agent is a live object the developer already
    has, not a credential to detect at boot.

    Wires the new provider into the same per-provider circuit breaker /
    rate-limiting / pruning / caching pipeline every other provider gets for
    free — `routers/generate.py` only ever looks providers and breakers up
    by name off `app.state`, so nothing there needs to know this provider
    isn't vendor-backed.
    """
    settings: Settings = app.state.settings
    provider = AgentProvider(name=name, agent=agent, cache_enabled=cache_enabled)
    app.state.providers[name] = provider
    app.state.circuit_breakers[name] = CircuitBreaker(
        failure_threshold=(
            circuit_breaker_failure_threshold
            if circuit_breaker_failure_threshold is not None
            else settings.circuit_breaker_failure_threshold
        ),
        reset_timeout_s=(
            circuit_breaker_reset_timeout_s
            if circuit_breaker_reset_timeout_s is not None
            else settings.circuit_breaker_reset_timeout_s
        ),
    )
    return provider


# --- request translation: neutral -> plain (role, content) message tuples ---


def _turn_text(parts: list[Part]) -> str:
    texts: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        else:
            raise ValueError(
                "agent provider is text-only for v1; binary parts are not supported (see issue #27)"
            )
    return "".join(texts)


def _to_agent_messages(request: NeutralGenerateRequest) -> list[tuple[str, str]]:
    """Turns/system -> `[(role, content), ...]` message tuples.

    This is the shape LangChain chat models and most `Runnable`s built from
    `ChatPromptTemplate` accept directly as `.invoke`/`.astream` input.
    """
    messages: list[tuple[str, str]] = []
    if request.system:
        messages.append(("system", _turn_text(request.system)))
    messages.extend((turn.role, _turn_text(turn.parts)) for turn in request.turns)
    return messages


# --- invocation: LangGraph CompiledStateGraph path (intermediate events) ---


def _is_langgraph_compiled_graph(agent: Any) -> bool:
    """True only for a compiled LangGraph graph; always False if `langgraph` isn't installed."""
    if CompiledStateGraph is None:
        return False
    return isinstance(agent, CompiledStateGraph)


async def _stream_langgraph_events(
    agent: Any, messages: list[tuple[str, str]]
) -> AsyncIterator[bytes]:
    """Stream a `CompiledStateGraph` via `stream_mode=["messages", "custom"]`.

    "messages"-mode chunks are `(message_chunk, metadata)` pairs (one per
    token/chunk a chat model inside a node streams); translated the same way
    the duck-typed path translates any chunk, via `_extract_text`. Metadata
    is discarded — the neutral contract has no slot for it.

    "custom"-mode chunks are whatever a node passed to `get_stream_writer()`.
    Only an `intermediate_step()` payload is recognized and forwarded; any
    other shape is a mistake the gateway won't guess the intent of, so it's
    dropped with a logged warning rather than passed through opaque or
    raised (a malformed emission from one node shouldn't kill a stream the
    rest of the graph is otherwise producing correctly).

    Note: an open LangGraph bug (langchain-ai/langgraph#6447) drops custom
    events emitted from *async tools* (as opposed to graph nodes) under
    `astream(stream_mode="custom")` — not worked around here.
    """
    stream = agent.astream({"messages": messages}, stream_mode=["messages", "custom"])
    async for mode, payload in stream:
        if mode == "messages":
            message_chunk, _metadata = payload
            text = _extract_text(message_chunk)
            if text:
                yield _text_event(text)
        elif mode == "custom":
            if isinstance(payload, _IntermediateStepPayload):
                yield _intermediate_event(payload)
            else:
                logger.warning(
                    "dropping langgraph custom stream payload not built with "
                    "intermediate_step(): %r",
                    payload,
                )


# --- invocation: duck-typed .astream -> .stream -> .ainvoke -> .invoke fallback chain ---


async def _stream_text(agent: Any, messages: list[tuple[str, str]]) -> AsyncIterator[str]:
    """Call whichever of the agent's methods exists, in order of streaming fidelity.

    `.astream` is preferred (native async streaming, one chunk in -> one
    chunk out). `.stream` (sync generator) is drained in a worker thread and
    its chunks re-yielded — a documented limitation of this fallback: a sync
    Runnable's chunks arrive as soon as produced from the thread's
    perspective, but the whole call still runs off the event loop, same as
    `.ainvoke`/`.invoke` below. `.ainvoke`/`.invoke` (no streaming support at
    all) each yield their one final result as a single chunk.
    """
    astream = getattr(agent, "astream", None)
    if astream is not None:
        async for chunk in astream(messages):
            text = _extract_text(chunk)
            if text:
                yield text
        return
    stream = getattr(agent, "stream", None)
    if stream is not None:
        chunks = await asyncio.to_thread(lambda: list(stream(messages)))
        for chunk in chunks:
            text = _extract_text(chunk)
            if text:
                yield text
        return
    ainvoke = getattr(agent, "ainvoke", None)
    if ainvoke is not None:
        text = _extract_text(await ainvoke(messages))
        if text:
            yield text
        return
    invoke = getattr(agent, "invoke", None)
    if invoke is not None:
        text = _extract_text(await asyncio.to_thread(invoke, messages))
        if text:
            yield text
        return
    raise TypeError(
        f"agent {agent!r} has none of .astream/.stream/.ainvoke/.invoke — "
        "not a Runnable-shaped object"
    )


def _extract_text(chunk: Any) -> str:
    """Best-effort text extraction from a chunk.

    Handles a bare string, a `.content` attribute (LangChain
    `AIMessage`/`AIMessageChunk`), or a `{"content": ...}` dict.
    """
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(chunk, dict):
        dict_content = chunk.get("content")
        if isinstance(dict_content, str):
            return dict_content
    return ""


# --- response translation: extracted text -> neutral SSE, one chunk -> one event ---


def _text_event(text: str) -> bytes:
    event = NeutralStreamEvent(delta=Delta(role="assistant", parts=[TextPart(text=text)]))
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n".encode()


def _final_event() -> bytes:
    event = NeutralStreamEvent(finish_reason=FinishReason.STOP)
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n".encode()


def _intermediate_event(step: "_IntermediateStepPayload") -> bytes:
    event = NeutralStreamEvent(intermediate=IntermediateStep(label=step.label, data=step.data))
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n".encode()
