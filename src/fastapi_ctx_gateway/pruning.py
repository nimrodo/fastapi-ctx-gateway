"""Context pruning: exact-duplicate turn removal + token-budget sliding window.

Budget-gated, not blanket-always-on: prune() is a byte-identical no-op
(same contents object returned) unless the conversation already exceeds
the model's configured cap. Only `text` parts are ever inspected — this
keeps the hot path cheap and means multimodal content is never at risk of
being silently corrupted by a heuristic that can't reason about it.
"""

from dataclasses import dataclass

from fastapi_ctx_gateway.config import TokenBudgetConfig
from fastapi_ctx_gateway.ratelimit import TokenEstimator
from fastapi_ctx_gateway.schemas.gemini import Content

__all__ = ["PruneResult", "TokenBudgetPruner", "prune_contents"]


@dataclass(frozen=True)
class PruneResult:
    """The outcome of a prune() call."""

    contents: list[Content]
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

    def prune(
        self, contents: list[Content], system_instruction: Content | None, model: str
    ) -> PruneResult:
        """Return contents unchanged unless the conversation exceeds the model's budget."""
        budget = self._token_budgets.budget_for(model)
        estimated = self._estimator.estimate(contents, system_instruction)
        if estimated <= budget:
            return PruneResult(contents=contents, pruned=False, dropped_turn_count=0)

        deduped, dedup_dropped = self._dedupe_exact(contents)
        windowed, window_dropped = self._sliding_window(deduped, budget, system_instruction)
        total_dropped = dedup_dropped + window_dropped
        return PruneResult(
            contents=windowed, pruned=total_dropped > 0, dropped_turn_count=total_dropped
        )

    def _dedupe_exact(self, contents: list[Content]) -> tuple[list[Content], int]:
        """Drop turns whose (role, text) exactly repeats an earlier turn."""
        seen: set[str] = set()
        kept: list[Content] = []
        dropped = 0
        for content in contents:
            fingerprint = self._fingerprint(content)
            if fingerprint in seen:
                dropped += 1
                continue
            seen.add(fingerprint)
            kept.append(content)
        return kept, dropped

    @staticmethod
    def _fingerprint(content: Content) -> str:
        # Text only, deliberately: hashing/comparing non-text bytes (images,
        # audio, files) would be slow and isn't needed for v1's exact-repeat
        # detection (e.g. a client retrying the same text turn).
        texts = [part.text for part in content.parts if part.text is not None]
        return content.role + "\x00" + "\x00".join(texts)

    def _sliding_window(
        self, contents: list[Content], budget: int, system_instruction: Content | None
    ) -> tuple[list[Content], int]:
        """Keep the most recent turns that fit under budget; drop oldest first.

        The single most recent turn always survives, even if it alone
        exceeds the budget — dropping everything would defeat the request.
        """
        system_tokens = (
            self._estimator.estimate([], system_instruction) if system_instruction else 0
        )
        kept: list[Content] = []
        running_tokens = system_tokens
        dropped = 0
        for content in reversed(contents):
            content_tokens = self._estimator.estimate([content], None)
            if kept and running_tokens + content_tokens > budget:
                dropped += 1
                continue
            kept.append(content)
            running_tokens += content_tokens
        kept.reverse()
        return kept, dropped


def prune_contents(
    contents: list[Content],
    system_instruction: Content | None,
    model: str,
    token_budgets: TokenBudgetConfig,
) -> PruneResult:
    """Convenience wrapper for one-off pruning without holding a TokenBudgetPruner."""
    return TokenBudgetPruner(token_budgets).prune(contents, system_instruction, model)
