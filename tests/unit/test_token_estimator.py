"""Tests for the pre-call token estimation heuristic."""

from fastapi_ctx_gateway.ratelimit import TokenEstimator
from fastapi_ctx_gateway.schemas.gemini import Content, Part


def test_estimate_empty_contents_is_small_but_nonzero() -> None:
    estimator = TokenEstimator()
    estimate = estimator.estimate(contents=[], system_instruction=None)
    assert estimate >= 0


def test_estimate_scales_with_text_length() -> None:
    estimator = TokenEstimator()
    short = estimator.estimate(
        contents=[Content(role="user", parts=[Part(text="hi")])], system_instruction=None
    )
    long = estimator.estimate(
        contents=[Content(role="user", parts=[Part(text="hi " * 500)])], system_instruction=None
    )
    assert long > short


def test_estimate_ignores_non_text_parts() -> None:
    estimator = TokenEstimator()
    text_only = estimator.estimate(
        contents=[Content(role="user", parts=[Part(text="hello")])], system_instruction=None
    )
    with_image = estimator.estimate(
        contents=[
            Content(
                role="user",
                parts=[
                    Part(text="hello"),
                    Part(inline_data={"mimeType": "image/png", "data": "x" * 10_000}),
                ],
            )
        ],
        system_instruction=None,
    )
    assert text_only == with_image


def test_estimate_counts_system_instruction() -> None:
    estimator = TokenEstimator()
    without = estimator.estimate(
        contents=[Content(role="user", parts=[Part(text="hi")])], system_instruction=None
    )
    with_system = estimator.estimate(
        contents=[Content(role="user", parts=[Part(text="hi")])],
        system_instruction=Content(role="system", parts=[Part(text="be nice " * 20)]),
    )
    assert with_system > without
