"""Tests for TokenBudgetPruner: exact-dup removal + token-budget sliding window."""

from fastapi_ctx_gateway.config import TokenBudgetConfig
from fastapi_ctx_gateway.pruning import TokenBudgetPruner
from fastapi_ctx_gateway.schemas.gemini import Content, Part

TINY_BUDGET = TokenBudgetConfig(budgets={"test-model": 15}, default=15)
ROOMY_BUDGET = TokenBudgetConfig(budgets={"test-model": 100_000}, default=100_000)


def _turn(text: str, role: str = "user") -> Content:
    return Content(role=role, parts=[Part(text=text)])


def test_under_budget_is_a_byte_identical_noop() -> None:
    pruner = TokenBudgetPruner(ROOMY_BUDGET)
    contents = [_turn("hi"), _turn("hello", role="model")]
    result = pruner.prune(contents, system_instruction=None, model="test-model")
    assert result.pruned is False
    assert result.dropped_turn_count == 0
    assert result.contents is contents  # identity, not just equality


def test_over_budget_removes_exact_duplicate_turns() -> None:
    pruner = TokenBudgetPruner(TINY_BUDGET)
    contents = [_turn("same"), _turn("same"), _turn("different one")]
    result = pruner.prune(contents, system_instruction=None, model="test-model")
    texts = [part.text for content in result.contents for part in content.parts]
    assert texts.count("same") == 1
    assert result.pruned is True


def test_over_budget_drops_oldest_turns_first_keeps_most_recent() -> None:
    pruner = TokenBudgetPruner(TINY_BUDGET)
    contents = [_turn("oldest " * 20), _turn("middle " * 20), _turn("newest")]
    result = pruner.prune(contents, system_instruction=None, model="test-model")
    texts = [part.text for content in result.contents for part in content.parts]
    assert texts[-1] == "newest"
    assert "oldest " * 20 not in texts


def test_most_recent_turn_always_survives_even_if_alone_over_budget() -> None:
    pruner = TokenBudgetPruner(TINY_BUDGET)
    contents = [_turn("way too much text " * 50)]
    result = pruner.prune(contents, system_instruction=None, model="test-model")
    assert len(result.contents) == 1


def test_unknown_model_falls_back_to_default_budget() -> None:
    config = TokenBudgetConfig(budgets={"test-model": 100_000}, default=20)
    pruner = TokenBudgetPruner(config)
    contents = [_turn("oldest " * 20), _turn("newest")]
    result = pruner.prune(contents, system_instruction=None, model="some-unlisted-model")
    assert result.pruned is True  # triggered by the tiny default cap, not test-model's large one


def test_non_text_parts_are_never_inspected_by_dedup() -> None:
    """Two turns with identical text but different inline data still dedupe as one.

    Dedup fingerprints text only (never non-text bytes) — deliberately, so
    it never has to hash/compare large binary blobs on the hot path.
    """
    pruner = TokenBudgetPruner(TINY_BUDGET)
    turn_a = Content(
        role="user",
        parts=[Part(text="same"), Part(inline_data={"mimeType": "image/png", "data": "AAAA"})],
    )
    turn_b = Content(
        role="user",
        parts=[Part(text="same"), Part(inline_data={"mimeType": "image/png", "data": "ZZZZ"})],
    )
    # The duplicates are the most recent turns, so at least one survives
    # windowing regardless of how tight the budget is.
    contents = [_turn("filler " * 10), turn_a, turn_b]
    result = pruner.prune(contents, system_instruction=None, model="test-model")
    dup_survivors = [c for c in result.contents if any(p.text == "same" for p in c.parts)]
    assert len(dup_survivors) == 1


def test_non_text_parts_are_never_mutated() -> None:
    pruner = TokenBudgetPruner(ROOMY_BUDGET)
    inline = {"mimeType": "image/png", "data": "AAAA"}
    turn = Content(role="user", parts=[Part(text="hi"), Part(inline_data=inline)])
    result = pruner.prune([turn], system_instruction=None, model="test-model")
    kept_inline = next(p.inline_data for c in result.contents for p in c.parts if p.inline_data)
    assert kept_inline == inline
