"""Tests for TokenBudgetPruner: exact-dup removal + token-budget sliding window."""

from fastapi_ctx_gateway.config import TokenBudgetConfig
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.schemas.neutral import BinaryPart, TextPart, Turn

TINY_BUDGET = TokenBudgetConfig(budgets={"test-model": 15}, default=15)
ROOMY_BUDGET = TokenBudgetConfig(budgets={"test-model": 100_000}, default=100_000)


def _turn(text: str, role: str = "user") -> Turn:
    return Turn(role=role, parts=[TextPart(text=text)])


def test_under_budget_is_a_byte_identical_noop() -> None:
    pruner = TokenBudgetPruner(ROOMY_BUDGET)
    turns = [_turn("hi"), _turn("hello", role="assistant")]
    result = pruner.prune(turns, system=None, model="test-model")
    assert result.pruned is False
    assert result.dropped_turn_count == 0
    assert result.turns is turns  # identity, not just equality


def test_over_budget_removes_exact_duplicate_turns() -> None:
    pruner = TokenBudgetPruner(TINY_BUDGET)
    turns = [_turn("same"), _turn("same"), _turn("different one")]
    result = pruner.prune(turns, system=None, model="test-model")
    texts = [part.text for turn in result.turns for part in turn.parts]
    assert texts.count("same") == 1
    assert result.pruned is True


def test_over_budget_drops_oldest_turns_first_keeps_most_recent() -> None:
    pruner = TokenBudgetPruner(TINY_BUDGET)
    turns = [_turn("oldest " * 20), _turn("middle " * 20), _turn("newest")]
    result = pruner.prune(turns, system=None, model="test-model")
    texts = [part.text for turn in result.turns for part in turn.parts]
    assert texts[-1] == "newest"
    assert "oldest " * 20 not in texts


def test_most_recent_turn_always_survives_even_if_alone_over_budget() -> None:
    pruner = TokenBudgetPruner(TINY_BUDGET)
    turns = [_turn("way too much text " * 50)]
    result = pruner.prune(turns, system=None, model="test-model")
    assert len(result.turns) == 1


def test_unknown_model_falls_back_to_default_budget() -> None:
    config = TokenBudgetConfig(budgets={"test-model": 100_000}, default=20)
    pruner = TokenBudgetPruner(config)
    turns = [_turn("oldest " * 20), _turn("newest")]
    result = pruner.prune(turns, system=None, model="some-unlisted-model")
    assert result.pruned is True  # triggered by the tiny default cap, not test-model's large one


def test_non_text_parts_are_never_inspected_by_dedup() -> None:
    """Two turns with identical text but different binary data still dedupe as one.

    Dedup fingerprints text only (never non-text bytes) — deliberately, so
    it never has to hash/compare large binary blobs on the hot path.
    """
    pruner = TokenBudgetPruner(TINY_BUDGET)
    turn_a = Turn(
        role="user",
        parts=[TextPart(text="same"), BinaryPart(mime_type="image/png", data="AAAA")],
    )
    turn_b = Turn(
        role="user",
        parts=[TextPart(text="same"), BinaryPart(mime_type="image/png", data="ZZZZ")],
    )
    # The duplicates are the most recent turns, so at least one survives
    # windowing regardless of how tight the budget is.
    turns = [_turn("filler " * 10), turn_a, turn_b]
    result = pruner.prune(turns, system=None, model="test-model")
    dup_survivors = [
        t
        for t in result.turns
        if any(isinstance(p, TextPart) and p.text == "same" for p in t.parts)
    ]
    assert len(dup_survivors) == 1


def test_non_text_parts_are_never_mutated() -> None:
    pruner = TokenBudgetPruner(ROOMY_BUDGET)
    binary = BinaryPart(mime_type="image/png", data="AAAA")
    turn = Turn(role="user", parts=[TextPart(text="hi"), binary])
    result = pruner.prune([turn], system=None, model="test-model")
    kept_binary = next(p for t in result.turns for p in t.parts if isinstance(p, BinaryPart))
    assert kept_binary == binary
