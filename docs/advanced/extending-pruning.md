# Extending pruning

[`TokenBudgetPruner`][fastapi_ctx_gateway.pruning.TokenBudgetPruner] implements exactly two strategies today — exact-duplicate removal and a token-budget sliding window — chosen deliberately because neither needs a model call and neither has a false-positive risk that could silently drop content the model needed. If you need a different strategy, this is the shape to extend.

## Internals

```python
class TokenBudgetPruner:
    def prune(self, contents, system_instruction, model) -> PruneResult: ...
    def _dedupe_exact(self, contents) -> tuple[list[Content], int]: ...
    def _sliding_window(
        self, contents, budget, system_instruction
    ) -> tuple[list[Content], int]: ...
```

`prune()` is budget-gated: it estimates tokens first and returns the input **unchanged** (same object, not just an equal copy) if the conversation is already under budget. Only once over budget does it call `_dedupe_exact` then `_sliding_window`, in that order.

Both internal methods only ever look at `text` parts of each `Content` turn, and both operate on whole turns — never on individual `Part`s within a turn. That invariant is load-bearing: it's what guarantees a multimodal turn (an image alongside text, say) is never partially mangled.

## Adding a strategy

A semantic near-duplicate collapser (turns that are similar but not identical) was deliberately deferred — it would need the same embedding infrastructure the gateway is already latency-constrained on, and carries real risk of dropping content the model needed. If you add one:

1. Keep it as an additional private method (`_collapse_near_duplicates`, say), called from `prune()` only once already over budget — don't make it part of the always-on path.
2. Preserve the whole-turn, text-only invariant unless you have a specific reason not to.
3. Return the dropped-turn count so `PruneResult.dropped_turn_count` (and the `prune_triggered_total` metric) stay accurate.

## Testing

`tests/unit/test_pruning.py` is the reference for how these are tested — a `TokenBudgetConfig` with an artificially tiny budget makes over-budget behavior easy to trigger without huge fixture strings. A new strategy should follow the same pattern: unit tests against a tiny budget, plus a router-level integration test (`tests/integration/test_stream_endpoint_pruning.py`) confirming the pruned payload is what actually reaches Gemini.
