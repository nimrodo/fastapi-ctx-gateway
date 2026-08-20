# Domain glossary

Vocabulary used consistently across this codebase, its tests, commit messages, and ADRs. When naming a concept in code, an issue, or a design proposal, use the term as defined here rather than a synonym.

**Provider** — a pluggable adapter (`fastapi_ctx_gateway.providers.Provider`) that translates the gateway's own neutral request/response contract to/from one upstream LLM API. `gemini` is always registered; `openai` registers only once its credentials are configured (`GATEWAY_OPENAI_API_KEY`). See ADR-0006.

**Tenant** — a caller of the gateway, authenticated via a gateway-issued API key (`x-gateway-api-key` header), resolved to a `Tenant(id, api_key)`. Distinct from a provider's own upstream API key (Gemini's, OpenAI's, ...), which the gateway holds once per provider and uses for every tenant.

**Hit path** — the code path taken when a semantically similar cached response is found: embed → Redis lookup → response synthesis, no call to the upstream provider. Target overhead: ≤15-20ms total. See `cache.hit_path` (OTel span).

**Miss path** — the code path taken when no cache hit occurs (cache disabled, ineligible, or a genuine miss): auth → rate-limit → breaker-check → prune, then proxy through to the configured provider. The gateway's own overhead on this path (everything before the first byte is proxied) targets low single-digit ms, additive to the provider's own latency. See `request.pre_proxy` (OTel span).

**Cache eligibility** — a request is eligible for the semantic cache only when its output is meant to be near-deterministic: no `tools` enabled, and `temperature` explicitly set at or below `cache_temperature_threshold` (default 0.3). An *unset* temperature is treated as "not explicitly low," not as permission to cache — see ADR-0003.

**Pruning budget** — the per-model token cap (`TokenBudgetConfig`) that gates context pruning. Pruning is a no-op (byte-identical passthrough) unless the conversation's estimated tokens exceed this cap; it is deliberately conservative relative to each model's real context window, since its purpose is triggering pruning early for latency/cache-key stability, not guarding against hard overflow.

**Fail-open** — the semantic cache's failure contract: any lookup/store error, timeout, or unreachable Redis at startup is treated as "cache disabled" or "this request is a miss," never as a request failure. The cache must never become a hard dependency for serving traffic. Contrast with **fail-closed** (not used anywhere in this gateway).

**Circuit breaker state** — `CLOSED` (normal), `OPEN` (short-circuiting requests without calling the provider, after `circuit_breaker_failure_threshold` consecutive failures), `HALF_OPEN` (a single trial request allowed after `circuit_breaker_reset_timeout_s`, deciding whether to close or reopen). Per-worker, per-provider, and in-memory — never a synchronous cross-worker check (that would reintroduce the round-trip cost the breaker exists to avoid), and never shared across providers (an outage on one upstream must not short-circuit an unrelated one). See ADR-0005, ADR-0006.

**Reconciliation** — the post-response correction of a tenant's token-budget counter, once the provider's real usage is known (translated into the neutral `Usage` type), against the pre-call estimate used for admission control. Runs after the stream has already fully reached the client, so it never delays what they see.

**Finished cleanly** — a stream is "finished cleanly" only once a chunk carrying `finish_reason` has been observed (`StreamAccumulator.finished_cleanly`). Anything else — client disconnect, upstream error, a truncated response — is not a clean finish, and neither reconciliation nor cache-store nor a circuit-breaker success is recorded for it.

**Pruned turns** — the `turns[]` array (neutral schema) after `TokenBudgetPruner.prune()` has run. The semantic cache always embeds and looks up against pruned turns, never raw — pruning happens first so equivalent conversations converge to the same cache key.
