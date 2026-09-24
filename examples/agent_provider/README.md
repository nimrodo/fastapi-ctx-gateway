# Example: registering your own agent as a provider

Demonstrates the pattern from [Registering your own agent](../../docs/tutorial/agent-provider.md):
`register_agent_provider()` puts a LangChain/LangGraph-shaped object of your own behind the
gateway's usual `/v1/{name}/{model}:streamGenerateContent` route, pipeline included.

Two providers are registered on the same app:

- `echo` — a hand-rolled duck-typed stub (only `.astream`, no LangChain import). Echoes your
  message back, one word at a time.
- `research` — a one-node LangGraph graph. The node emits a custom `intermediate_step()` event
  before streaming a canned reply, so you can see the `intermediate` SSE field on the wire.

Both are deterministic (backed by `GenericFakeChatModel`/a plain generator, not a real LLM), so
this example needs no LLM API key of its own.

## Run it

```bash
# Redis is only needed once a request actually reaches the gateway
# (rate limiting, pruning). The app boots fine without it.
docker compose up -d redis

# create_app() refuses to boot with zero configured providers, and doesn't
# know about agent providers registered after the fact — set a placeholder
# vendor key so it boots; nothing here ever calls Gemini.
export GATEWAY_GEMINI_UPSTREAM_KEY=unused

# Maps gateway-issued keys (what clients send in x-gateway-api-key) to a
# tenant id. Required — with no entries, every request gets 401.
export GATEWAY_TENANT_API_KEYS='{"my-gateway-key":"local-dev"}'

uv run uvicorn examples.agent_provider.app:app --app-dir . --reload
```

Then, from another terminal:

```bash
curl -N -X POST http://localhost:8000/v1/echo/default:streamGenerateContent \
  -H "x-gateway-api-key: my-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{"turns": [{"role": "user", "parts": [{"type": "text", "text": "hello there"}]}]}'
```

```
data: {"delta":{"role":"assistant","parts":[{"type":"text","text":"hello "}]}}

data: {"delta":{"role":"assistant","parts":[{"type":"text","text":"there "}]}}

data: {"finish_reason":"stop"}
```

```bash
curl -N -X POST http://localhost:8000/v1/research/default:streamGenerateContent \
  -H "x-gateway-api-key: my-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]}'
```

```
data: {"intermediate":{"label":"tool_call","data":{"tool":"search","query":"fastapi-ctx-gateway"}}}

data: {"delta":{"role":"assistant","parts":[{"type":"text","text":"Found 3 relevant results."}]}}

data: {"finish_reason":"stop"}
```

The `intermediate` event always arrives before the text it precedes — the node emits it, then
streams its reply.
