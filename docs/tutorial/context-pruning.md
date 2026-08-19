# Context pruning

Long-running conversations grow without bound unless something trims them. The gateway prunes `turns[]` — but only when a conversation actually needs it.

## Budget-gated, not always-on

[`TokenBudgetPruner.prune()`][fastapi_ctx_gateway.pruning.TokenBudgetPruner.prune] estimates the conversation's token count and compares it against the model's configured budget. **Under budget, the conversation passes through byte-identical** — no dedup, no trimming, no behavior change for well-behaved clients. Only once a conversation exceeds its budget does pruning actually do anything:

1. **Exact-duplicate turn removal** — a turn whose `(role, text)` exactly repeats an earlier turn is dropped (e.g. a client retrying the same message).
2. **Token-budget sliding window** — oldest turns are dropped first, keeping the system instruction and the most recent turns that fit. The single most recent turn always survives, even if it alone exceeds the budget.

Both operations only ever look at `text` parts — non-text parts (images, audio, files) are never inspected, never hashed, and whole turns are dropped rather than individual parts, so a multimodal turn is either kept intact or removed intact.

## Per-model budgets

```bash
GATEWAY_TOKEN_BUDGETS='{"budgets": {"gemini-3.7-flash": 32000, "gemini-2.5-pro": 64000}, "default": 16000}'
```

These are deliberately conservative relative to each model's *real* context window — they exist to trigger pruning early enough to matter for latency and cache-key stability, not to guard against a hard context-window overflow.

## Where it runs

Pruning happens after the rate-limit and circuit-breaker checks (so a request that's about to be rejected doesn't pay for it) but before the semantic-cache lookup — cached responses are always keyed against **pruned** contents, which increases the cache hit rate: two conversations that differ only in what pruning would've dropped anyway still land on the same cache key.

See [`TokenBudgetPruner`][fastapi_ctx_gateway.pruning.TokenBudgetPruner] for the full reference, or [Extending pruning](../advanced/extending-pruning.md) to add a new strategy.
