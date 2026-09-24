# fastapi-ctx-gateway

A **context-aware agentic API gateway** for LLM APIs (Gemini, OpenAI, and Anthropic today). It sits between clients and the upstream provider, handling stream proxying, semantic caching, token-aware rate limiting, and context pruning at the edge — with a target overhead budget of **~15-20ms on a cache-hit path**.

```bash
uv add fastapi-ctx-gateway
```

## Why

Calling an LLM provider directly from every client leaves real savings on the table: repeated or near-duplicate prompts re-run the full model every time, conversations grow unbounded until they blow past context windows and cost, and nothing protects your own capacity or the provider's quota when things go wrong. `fastapi-ctx-gateway` sits in the request path and handles all of that transparently, behind one provider-agnostic contract.

## Features

- **Stream proxying** — one neutral request/response contract (`turns`/`parts`) that a pluggable [`Provider`][fastapi_ctx_gateway.providers.base.Provider] translates to/from each upstream API. Gemini, OpenAI, and Anthropic ship today, each an opt-in dependency extra with none mandatory; see [ADR-0006](adr/0006-neutral-schema-and-provider-abstraction.md) and [ADR-0008](adr/0008-providers-are-opt-in-extras.md).
- **Agent provider** — wrap your own in-process LangChain `Runnable` or LangGraph `CompiledStateGraph` as a provider via `register_agent_provider()`, no vendor HTTP call or API key involved; see the [agent-provider tutorial](tutorial/agent-provider.md).
- **Semantic caching** — near-duplicate prompts hit a Redis-backed vector cache instead of calling the provider again. Tenant- and model-partitioned, fails open on any backend error.
- **Token-aware rate limiting** — TPM/RPM budgets per tenant and model, enforced with a single atomic Redis round trip.
- **Context pruning** — exact-duplicate turns dropped and a token-budget sliding window applied, but only once a conversation actually exceeds its budget. Untouched otherwise.
- **Circuit breaker** — a failing upstream trips its own per-provider, per-worker breaker so the gateway (and that upstream) aren't hammered during an outage; an outage on one provider never affects another.
- **Observability** — Prometheus counters and OpenTelemetry spans for every fallback and degradation decision, all on one `/metrics` endpoint.

## A minimal example

```python
from fastapi_ctx_gateway import create_app, Settings

app = create_app(Settings())
```

```bash
curl -X POST http://localhost:8000/v1/gemini/gemini-3.7-flash:streamGenerateContent \
  -H "x-gateway-api-key: your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{"turns": [{"role": "user", "parts": [{"type": "text", "text": "Hello!"}]}]}'
```

Head to the [Tutorial](tutorial/index.md) to get a gateway running end to end, or straight to the [Reference](reference/index.md) if you already know what you're looking for.

## Requirements

- Python 3.13+
- Redis Stack (the RediSearch/VSS module — used for rate limiting and, optionally, the semantic cache)
- At least one vendor provider configured — a Gemini, OpenAI, or Anthropic API key; this boot check runs before an [agent provider](tutorial/agent-provider.md) can be registered, so an agent-only deployment still needs one placeholder vendor key — see [ADR-0008](adr/0008-providers-are-opt-in-extras.md)

## Two latency budgets

Every request is measured against one of two separate budgets:

| Path | What it covers | Target |
|---|---|---|
| **Cache hit** | embed + Redis vector lookup + response synthesis, no provider call | ≤15-20ms total |
| **Cache miss** | auth + rate-limit check + prune, before the provider call | low single-digit ms, additive to the provider's own latency |

See [Concepts & glossary](concepts.md) for the full request lifecycle and vocabulary, and [Design decisions](adr/0001-gemini-classic-api-over-interactions-api.md) for why each major architectural fork was decided the way it was.
