# Registering your own agent

`GeminiProvider`, `OpenAIProvider`, and `AnthropicProvider` all speak HTTP to a vendor upstream. The **agent provider** is different: it wraps an in-process Python object you've already built — a LangChain `Runnable`, a LangGraph `CompiledStateGraph`, or anything shaped like one — and puts it behind the same route shape and pipeline (rate limiting, pruning, circuit breaker, cache) every other provider gets. There's no wire format to translate and no upstream credential, so it isn't registered via `Settings`/env vars like the vendor providers — you call [`register_agent_provider()`][fastapi_ctx_gateway.providers.agent.register_agent_provider] directly against the app object, any time after [`create_app()`][fastapi_ctx_gateway.app.create_app] returns:

```python
from fastapi_ctx_gateway import Settings, create_app
from fastapi_ctx_gateway.providers.agent import register_agent_provider

app = create_app(Settings())
register_agent_provider(app, name="my-agent", agent=my_runnable_or_graph)
```

`my_runnable_or_graph` is duck-typed: anything exposing `.astream`/`.stream`/`.ainvoke`/`.invoke` works, in that order of preference (see [`Provider`][fastapi_ctx_gateway.providers.base.Provider]). A `BinaryPart` (image, audio, or other binary content) in a request is translated to a LangChain content block — inline `data` becomes a `source_type: "base64"` block, a `uri` becomes `source_type: "url"` — and passed straight to `my_runnable_or_graph`; there's no capability detection, so whether it's actually handled depends on what the wrapped agent/model supports (an unsupported modality surfaces however the underlying call fails, translated to a neutral error event like any other agent exception).

!!! note "`create_app()` still needs one vendor provider configured"
    Registering an agent provider doesn't exempt you from `create_app()`'s "at least one provider must be configured" boot check (see [ADR-0008](../adr/0008-providers-are-opt-in-extras.md)) — that check runs before `register_agent_provider()` even has a chance to add yours. If your app has no other reason to talk to Gemini/OpenAI/Anthropic, set one placeholder vendor key anyway (e.g. `GATEWAY_GEMINI_UPSTREAM_KEY=unused`); nothing routes to it unless a client actually calls `/v1/gemini/...`.

## Route shape

Same as every other provider: `POST /v1/{name}/{model}:streamGenerateContent`, where `{name}` is whatever you passed to `register_agent_provider()`. `{model}` is accepted but ignored — one registration binds to one already-built agent object, not a family of selectable models.

=== "curl"

    ```bash
    curl -N -X POST \
      http://localhost:8000/v1/my-agent/default:streamGenerateContent \
      -H "x-gateway-api-key: my-gateway-key" \
      -H "Content-Type: application/json" \
      -d '{
        "turns": [
          {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
        ]
      }'
    ```

=== "Python (httpx)"

    ```python
    import httpx

    with httpx.stream(
        "POST",
        "http://localhost:8000/v1/my-agent/default:streamGenerateContent",
        headers={"x-gateway-api-key": "my-gateway-key"},
        json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
    ) as response:
        for chunk in response.iter_bytes():
            print(chunk.decode(), end="")
    ```

## The plain-`Runnable` path

Anything without `.astream` falls back to `.stream` (drained in a worker thread), then `.ainvoke`/`.invoke` (no streaming — the whole result arrives as one chunk). Each upstream chunk's text is extracted from a bare string, a `.content` attribute (LangChain `AIMessage`/`AIMessageChunk`), or a `{"content": ...}` dict, and translated 1:1 into a neutral `delta` event — same as a vendor provider's SSE bytes.

By default the semantic cache is **off** for agent providers (`cache_enabled=False`), unlike the vendor providers (`True`). An agent's own code can have side effects — tool calls, writes — that aren't safe to silently skip by replaying a cached response. Pass `cache_enabled=True` to `register_agent_provider()` to opt back in for a purely-functional agent.

## The LangGraph intermediate-events path

Register a compiled LangGraph graph (`CompiledStateGraph`) and the adapter switches to `stream_mode=["messages", "custom"]` automatically — no separate flag. `"messages"`-mode chunks (tokens streamed by a chat model called inside a node) translate to `delta` events exactly like the plain-Runnable path. `"custom"`-mode chunks are where a node can surface an arbitrary intermediate step:

```python
from langgraph.config import get_stream_writer
from fastapi_ctx_gateway.providers.agent import intermediate_step


async def my_node(state):
    writer = get_stream_writer()
    writer(intermediate_step(label="tool_call", data={"tool": "search", "query": "..."}))
    ...
```

`intermediate_step()` is the *only* shape the adapter recognizes on a `"custom"` chunk — a `writer()` payload built any other way (a bare dict, say) is dropped with a logged warning rather than guessed at or passed through opaque.

### What comes back

The same SSE stream every provider produces, with one addition: an `intermediate` field carrying whatever `label`/`data` the node passed, always ahead of the text it precedes:

```
data: {"intermediate":{"label":"tool_call","data":{"tool":"search","query":"..."}}}

data: {"delta":{"role":"assistant","parts":[{"type":"text","text":"Found 3 relevant results."}]}}

data: {"finish_reason":"stop"}
```

`data` is opaque to the gateway — whatever your node passed comes back verbatim; the gateway never interprets it as a tool-call representation of its own.

!!! warning "A known LangGraph limitation"
    [langchain-ai/langgraph#6447](https://github.com/langchain-ai/langgraph/issues/6447) drops custom events emitted from *async tools* (as opposed to graph nodes) under `astream(stream_mode="custom")`. Not worked around here — emit `intermediate_step()` from a node, not from inside a tool call, until that's fixed upstream.

## No pre-stream retry

Every vendor provider retries once before any bytes reach the client (see [Adding a provider](../advanced/adding-a-provider.md)). The agent provider deliberately doesn't: retrying arbitrary agent code risks re-triggering a side effect (a tool call, a write) the failed attempt already caused.

## Runnable example

A complete, dependency-light example covering both paths above — a hand-rolled duck-typed stub for the plain-`Runnable` path and a one-node LangGraph graph for the intermediate-events path, both deterministic and needing no API key — lives in
[`examples/agent_provider/`](https://github.com/nimrodo/fastapi-ctx-gateway/tree/main/examples/agent_provider).

Next: [Adding a provider](../advanced/adding-a-provider.md), if you're instead wiring up a *vendor* HTTP upstream rather than wrapping your own in-process agent.
