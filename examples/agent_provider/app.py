"""Example: registering your own LangChain/LangGraph object as a gateway provider.

Registers two agent providers on the same app, one per path the adapter
supports:

- `echo` — a hand-rolled, duck-typed stub (only `.astream`). No LangChain
  import at all: `register_agent_provider()` doesn't care what built the
  object, only that it's shaped like a `Runnable`.
- `research` — a compiled LangGraph graph whose one node emits a custom
  `intermediate_step()` event (surfaced on the wire as the `intermediate`
  SSE field) before streaming its reply.

Both are deterministic and need no external API key or network call, so this
example is runnable as-is.

Run with:

    uv run uvicorn examples.agent_provider.app:app --app-dir . --reload

See README.md in this directory for setup and curl examples.
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fastapi_ctx_gateway import Settings, create_app
from fastapi_ctx_gateway.providers.agent import intermediate_step, register_agent_provider


class EchoAgent:
    """Duck-typed Runnable stub: echoes the last user message back, one word at a time.

    Only implements `.astream` — no base class, no LangChain import — to
    demonstrate that the agent provider works with *anything* shaped like a
    Runnable, not just real LangChain objects.
    """

    async def astream(self, messages: list[tuple[str, str]]) -> AsyncIterator[str]:
        """Yield the last user message back, one word at a time."""
        _role, last_user_text = messages[-1]
        for word in last_user_text.split():
            yield f"{word} "


def _build_research_graph() -> CompiledStateGraph:
    """A one-node LangGraph graph demonstrating the intermediate-events path.

    The node emits a custom `intermediate_step()` event via
    `get_stream_writer()` (e.g. "I'm about to search for X") before its reply
    streams — see docs/tutorial/agent-provider.md for what this looks like on
    the wire. `GenericFakeChatModel` keeps the reply deterministic and
    key-free; swap it for a real chat model in your own graph.
    """

    async def research(state: MessagesState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(
            intermediate_step(
                label="tool_call", data={"tool": "search", "query": "fastapi-ctx-gateway"}
            )
        )
        model = GenericFakeChatModel(
            messages=iter([AIMessage(content="Found 3 relevant results.")])
        )
        chunks = [chunk async for chunk in model.astream(state["messages"])]
        return {"messages": chunks}

    builder = StateGraph(MessagesState)
    builder.add_node("research", research)
    builder.add_edge(START, "research")
    builder.add_edge("research", END)
    return builder.compile()


# The agent provider needs no vendor extra or credential of its own, but
# create_app() still refuses to boot with zero *registered* providers (see
# app.py::create_app) — it doesn't know about agent providers registered
# after the fact. Settings() below picks up GATEWAY_GEMINI_UPSTREAM_KEY from
# the environment purely to satisfy that boot check; nothing here ever routes
# to `/v1/gemini/...`, so a placeholder value works fine. See README.md.
app = create_app(Settings())
register_agent_provider(app, name="echo", agent=EchoAgent())
register_agent_provider(app, name="research", agent=_build_research_graph())
