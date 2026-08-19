# ADR-0001: Target Gemini's classic stateless API, not the Interactions API

## Context

Google's Gemini API offers two surfaces: the classic, stateless `generateContent`/`streamGenerateContent` endpoints (the client resends the full `contents[]` array on every call), and the newer Interactions API, which supports server-side conversation state (`store=true`, `previous_interaction_id`) so the client only sends the new turn.

## Decision

Target the classic stateless API only.

## Consequences

The gateway's core value proposition — pruning and caching against the full conversation context — requires seeing that full context on every request. If Google holds conversation state server-side, the client stops sending full history, and there is nothing left in the request for the gateway to prune or cache against. Adopting the Interactions API would require the gateway itself to become the state store, which is a materially different (and out of scope) product. Revisit only if a future requirement makes the gateway the authoritative conversation store.
