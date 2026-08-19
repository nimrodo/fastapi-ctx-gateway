"""Tests for the pre-call token estimation heuristic."""

from fastapi_ctx_gateway.ratelimit import TokenEstimator
from fastapi_ctx_gateway.schemas.neutral import BinaryPart, TextPart, Turn


def test_estimate_empty_turns_is_small_but_nonzero() -> None:
    estimator = TokenEstimator()
    estimate = estimator.estimate(turns=[], system=None)
    assert estimate >= 0


def test_estimate_scales_with_text_length() -> None:
    estimator = TokenEstimator()
    short = estimator.estimate(turns=[Turn(role="user", parts=[TextPart(text="hi")])], system=None)
    long = estimator.estimate(
        turns=[Turn(role="user", parts=[TextPart(text="hi " * 500)])], system=None
    )
    assert long > short


def test_estimate_ignores_non_text_parts() -> None:
    estimator = TokenEstimator()
    text_only = estimator.estimate(
        turns=[Turn(role="user", parts=[TextPart(text="hello")])], system=None
    )
    with_image = estimator.estimate(
        turns=[
            Turn(
                role="user",
                parts=[
                    TextPart(text="hello"),
                    BinaryPart(mime_type="image/png", data="x" * 10_000),
                ],
            )
        ],
        system=None,
    )
    assert text_only == with_image


def test_estimate_counts_system() -> None:
    estimator = TokenEstimator()
    without = estimator.estimate(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])], system=None
    )
    with_system = estimator.estimate(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        system=[TextPart(text="be nice " * 20)],
    )
    assert with_system > without
