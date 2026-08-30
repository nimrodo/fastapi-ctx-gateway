# ADR-0007: Build the Anthropic provider on the official Claude Python SDK

## Context

`GeminiProvider` and `OpenAIProvider` are hand-written raw-HTTP wire adapters: they borrow the lifespan's pooled `httpx.AsyncClient`, POST to the upstream, and split the response SSE bytes 1:1 into neutral events with `providers/sse.py`. `docs/advanced/adding-a-provider.md` codifies that shape as *the* way to add a provider.

Anthropic ships a mature official Python SDK (`anthropic`). Adding a third provider raised the question of whether to keep hand-rolling raw HTTP against `POST /v1/messages` for consistency, or use the SDK.

## Decision

Build `AnthropicProvider` (`fastapi_ctx_gateway.providers.anthropic`) on `anthropic.AsyncAnthropic`. The SDK is an **opt-in dependency** — the `[anthropic]` extra (see [ADR-0008](0008-providers-are-opt-in-extras.md)) — not a core dependency.

The adapter still honours the `Provider` contract exactly: neutral request in, neutral SSE bytes out, `stream()` never raises. Only the internals differ from the raw-HTTP adapters.

## Consequences

- **Typed SDK events replace `providers/sse.py` byte framing.** `client.messages.create(..., stream=True)` yields typed event objects (`message_start`, `content_block_delta`, `message_delta`, ...), so `iter_sse_data_lines` is unused here — only `neutral_error_event` is still shared. Translation is event-object → `NeutralStreamEvent`, one event to zero or one neutral event (structural events like `content_block_start` / `ping` / `message_stop` are swallowed, as `OpenAIProvider` swallows `[DONE]`).
- **The provider synthesizes `total_tokens`.** Anthropic splits token accounting across events: `message_start` carries prompt tokens (plus cache-read/creation tokens), `message_delta` carries completion tokens and the stop reason. The adapter holds the prompt count and emits one fully-populated `Usage` (prompt/completion/total, computed) on the `message_delta` event — the router reconciles rate-limit against `usage.total_tokens`, which the SDK never sends directly.
- **SDK retry replaces the manual pre-stream loop.** The injected client is built with `max_retries=1`. For a streaming call the SDK's retry is entirely pre-first-event, which is exactly the "retry only before any bytes reached the client" policy the raw-HTTP adapters implement by hand with `_MAX_RETRIES`. `stream()` still wraps the SDK calls in `try/except` (typed `anthropic.APIError` subclasses) to translate a post-retry failure into a terminal `neutral_error_event` and keep the never-raise contract. A mid-stream transport drop is never retried — it just emits the terminal error.
- **The SDK client is injected, and owns its own `httpx2` pool.** `AnthropicProvider.__init__(client, default_max_tokens)` takes the client; `AnthropicProvider.from_settings(...)` builds a real one for production (called by the provider registry). The SDK bundles `httpx2`, so it cannot share the gateway's `httpx.AsyncClient` — `AnthropicProvider.aclose()` closes the SDK client and `app.py`'s lifespan calls it on shutdown, alongside the Redis client.
- **Tests mock the injected client, not the HTTP call.** `respx` patches `httpx`, not `httpx2`, so the repo's usual "respx the upstream" seam doesn't reach the SDK. Response-translation unit tests feed canned SDK-shaped event objects straight into the translator (no HTTP); the router-level integration test swaps `app.state.providers["anthropic"]` for an `AnthropicProvider` wrapping a stub client. `adding-a-provider.md`'s respx guidance applies to raw-HTTP adapters only.
- **Anthropic's mandatory `max_tokens`.** The Messages API requires `max_tokens`; the neutral `GenerationConfig.max_output_tokens` is optional. `GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS` (default 4096) is the fallback when a request omits it.
- **`top_k` is forwarded** (Anthropic supports it), unlike the Gemini/OpenAI adapters which drop it. `candidate_count` and `safety_settings` are dropped (no equivalent). `tools`/`tool_config` stay opaque pass-through per ADR-0006. A `BinaryPart` in `system` is dropped — Anthropic's `system` is text-only.
