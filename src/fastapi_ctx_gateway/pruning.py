"""Context pruning: exact-duplicate turn removal + token-budget sliding window.

Budget-gated, not blanket-always-on: prune() is a byte-identical no-op
(same turns object returned) unless the conversation already exceeds
the model's configured cap. Only `text` parts are ever inspected — this
keeps the hot path cheap and means multimodal content is never at risk of
being silently corrupted by a heuristic that can't reason about it.
"""

from pydantic import BaseModel

from fastapi_ctx_gateway.config import TokenBudgetConfig
from fastapi_ctx_gateway.ratelimit import TokenEstimator
from fastapi_ctx_gateway.schemas.neutral import Part, TextPart, Turn

__all__ = ["PruneResult", "TokenBudgetPruner", "prune_turns"]


class PruneResult(BaseModel):
    """The outcome of a prune() call."""

    turns: list[Turn]
    pruned: bool
    dropped_turn_count: int


class TokenBudgetPruner:
    """Drops whole turns (never individual parts) to fit a per-model token budget."""

    def __init__(
        self, token_budgets: TokenBudgetConfig, estimator: TokenEstimator | None = None
    ) -> None:
        """Wire in the budget table and (optionally) a shared TokenEstimator."""
        self._token_budgets = token_budgets
        self._estimator = estimator or TokenEstimator()

    def prune(self, turns: list[Turn], system: list[Part] | None, model: str) -> PruneResult:
        """Return turns unchanged unless the conversation exceeds the model's budget."""
        budget = self._token_budgets.budget_for(model)
        estimated = self._estimator.estimate(turns, system)
        if estimated <= budget:
            # model_construct (not the normal constructor) so the returned
            # `turns` is the exact same list object, not a validation copy —
            # prune() must be a byte-identical no-op under budget.
            return PruneResult.model_construct(turns=turns, pruned=False, dropped_turn_count=0)

        deduped, dedup_dropped = self._dedupe_exact(turns)
        windowed, window_dropped = self._sliding_window(deduped, budget, system)
        total_dropped = dedup_dropped + window_dropped
        return PruneResult(
            turns=windowed, pruned=total_dropped > 0, dropped_turn_count=total_dropped
        )

    def _dedupe_exact(self, turns: list[Turn]) -> tuple[list[Turn], int]:
        """Drop turns whose (role, text) exactly repeats an earlier turn."""
        seen: set[str] = set()
        kept: list[Turn] = []
        dropped = 0
        for turn in turns:
            fingerprint = self._fingerprint(turn)
            if fingerprint in seen:
                dropped += 1
                continue
            seen.add(fingerprint)
            kept.append(turn)
        return kept, dropped

    @staticmethod
    def _fingerprint(turn: Turn) -> str:
        # Text only, deliberately: hashing/comparing non-text bytes (images,
        # audio, files) would be slow and isn't needed for v1's exact-repeat
        # detection (e.g. a client retrying the same text turn).
        texts = [part.text for part in turn.parts if isinstance(part, TextPart)]
        return turn.role + "\x00" + "\x00".join(texts)

    def _sliding_window(
        self, turns: list[Turn], budget: int, system: list[Part] | None
    ) -> tuple[list[Turn], int]:
        """Keep the most recent turns that fit under budget; drop oldest first.

        The single most recent turn always survives, even if it alone
        exceeds the budget — dropping everything would defeat the request.
        """
        system_tokens = self._estimator.estimate([], system) if system else 0
        kept: list[Turn] = []
        running_tokens = system_tokens
        dropped = 0
        for turn in reversed(turns):
            turn_tokens = self._estimator.estimate([turn], None)
            if kept and running_tokens + turn_tokens > budget:
                dropped += 1
                continue
            kept.append(turn)
            running_tokens += turn_tokens
        kept.reverse()
        return kept, dropped


def prune_turns(
    turns: list[Turn],
    system: list[Part] | None,
    model: str,
    token_budgets: TokenBudgetConfig,
) -> PruneResult:
    """Convenience wrapper for one-off pruning without holding a TokenBudgetPruner."""
    return TokenBudgetPruner(token_budgets).prune(turns, system, model)
