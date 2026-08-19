# fastapi-ctx-gateway

A **context-aware agentic API gateway** for Google's Gemini API. It sits between clients and Gemini, handling stream proxying, semantic caching, token-aware rate limiting, and context pruning at the edge — with a target overhead budget of **~15-20ms on a cache-hit path**.

```bash
uv add fastapi-ctx-gateway
```

## Why

Calling an LLM provider directly from every client leaves real savings on the table: repeated or near-duplicate prompts re-run the full model every time, conversations grow unbounded until they blow past context windows and cost, and nothing protects your own capacity or the provider's quota when things go wrong. `fastapi-ctx-gateway` sits in the request path and handles all of that transparently, without changing the wire protocol your clients already speak.

## Features

- **Stream proxying** — Gemini's native `contents`/`parts` wire schema, unchanged. No translation layer, no SDK lock-in.
- **Semantic caching** — near-duplicate prompts hit a Redis-backed vector cache instead of calling Gemini again. Tenant- and model-partitioned, fails open on any backend error.
- **Token-aware rate limiting** — TPM/RPM budgets per tenant and model, enforced with a single atomic Redis round trip.
- **Context pruning** — exact-duplicate turns dropped and a token-budget sliding window applied, but only once a conversation actually exceeds its budget. Untouched otherwise.
- **Circuit breaker** — a failing Gemini backend trips a per-worker breaker so the gateway (and Gemini) aren't hammered during an outage.
- **Observability** — Prometheus counters and OpenTelemetry spans for every fallback and degradation decision, all on one `/metrics` endpoint.

## A minimal example

```python
from fastapi_ctx_gateway import create_app, Settings

app = create_app(Settings())
```

```bash
curl -X POST http://localhost:8000/v1/gemini-3.7-flash:streamGenerateContent \
  -H "x-gateway-api-key: your-gateway-key" \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"role": "user", "parts": [{"text": "Hello!"}]}]}'
```

Head to the [Tutorial](tutorial/index.md) to get a gateway running end to end, or straight to the [Reference](reference/index.md) if you already know what you're looking for.

## Requirements

- Python 3.13+
- Redis Stack (the RediSearch/VSS module — used for rate limiting and, optionally, the semantic cache)
- A Gemini API key

## Two latency budgets

Every request is measured against one of two separate budgets:

| Path | What it covers | Target |
|---|---|---|
| **Cache hit** | embed + Redis vector lookup + response synthesis, no Gemini call | ≤15-20ms total |
| **Cache miss** | auth + rate-limit check + prune, before the Gemini call | low single-digit ms, additive to Gemini's own latency |

See [Concepts & glossary](concepts.md) for the full request lifecycle and vocabulary, and [Design decisions](adr/0001-gemini-classic-api-over-interactions-api.md) for why each major architectural fork was decided the way it was.
