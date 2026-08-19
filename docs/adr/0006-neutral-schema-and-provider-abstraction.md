# ADR-0006: Neutral request/response schema and a pluggable provider abstraction

## Context

The gateway originally proxied Gemini's own `streamGenerateContent` wire schema unchanged — no translation layer, by design. That decision made sense for a single-provider gateway, but it meant every internal component (the request type on the route itself, the pruner, the token estimator, the semantic-cache key/eligibility logic) was directly coupled to Gemini's `Content`/`Part` shapes. Supporting a second LLM provider (e.g. OpenAI) on top of that coupling would have meant either duplicating the entire pipeline per provider, or threading provider-specific branches through code that has no business knowing which provider it's serving.

## Decision

Introduce the gateway's own **neutral** request/response/error contract (`fastapi_ctx_gateway.schemas.neutral`) — not modeled on Gemini's or any other provider's wire shape — as the gateway's actual public contract. A pluggable `Provider` interface (`fastapi_ctx_gateway.providers.base.Provider`) translates between this neutral contract and one upstream API; `GeminiProvider` is the only implementation today, with Gemini's own wire schema demoted to an adapter-internal detail (`fastapi_ctx_gateway.providers.gemini_wire`, moved out of `schemas/`).

The single streaming route moves from `POST /v1/{model}:streamGenerateContent` to `POST /v1/{provider}/{model}:streamGenerateContent`, with the request body and streamed SSE events now neutral-schema. This is an explicit, accepted breaking change — the repo is pre-release with no external consumers, so there is no legacy route or dual-accept shim; docs, the example app, and the test suite were migrated in the same change.

The pruner, token estimator, and cache-key/eligibility logic were rewritten to operate on the neutral schema instead of Gemini's types — mechanical retypes with unchanged algorithms. `RateLimiter` and `CircuitBreaker` needed no changes; they were already schema-agnostic.

Errors are normalized too: a single `NeutralErrorEvent` envelope (`{"error": {"message", "type", "provider_status"}}`) replaces the old ad hoc Gemini-shaped synthetic error event, and the existing rate-limit (429) / circuit-breaker (503) JSON error bodies were reshaped to match — one error shape everywhere in the API, not one for streamed provider failures and another for gateway-side rejections.

## Consequences

- **Translation cost on the streaming hot path.** Because forwarding now requires interpreting each upstream SSE event to translate it, `GeminiProvider` buffers only up to the next SSE event boundary (never the whole response) before forwarding — one upstream event still translates to exactly one neutral event, preserving the low-latency streaming characteristics, at the cost of a small, bounded (one event's worth) amount of buffering that didn't exist under pure byte passthrough.
- **`tools`/`tool_config`/`safety_settings` stay opaque pass-through** in the neutral schema (`dict[str, Any]`) rather than being modeled and translated per-provider — real per-provider tool-calling translation is deferred to whichever future provider actually needs it.
- **The bounded pre-stream retry moved into the provider.** `Provider.stream()` never raises for upstream/transport failures — those are translated in-band to a terminal error event — so the one-retry-before-any-bytes policy that used to live in the generic `stream_with_retry` wrapper now lives inside `GeminiProvider` itself. The generic wrapper's own retry-on-exception path remains as a safety net for failures the adapter doesn't catch, but rarely fires in practice now.
- **Every existing doc, README, and test that assumed Gemini's native wire format was the gateway's contract needed updating** in this same change — there was no way to land the schema change incrementally without a transient inconsistency, given the "no dual-accept" decision.
- **Adding a second provider (e.g. OpenAI) is now additive**, not a parallel pipeline: a new `Provider` implementation plus a registry entry in `app.py`, with the pruner/estimator/cache/rate-limiter/breaker all reused unchanged.
