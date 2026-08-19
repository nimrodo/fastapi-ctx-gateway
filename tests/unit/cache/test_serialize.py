"""Tests for canonicalize_turns: deterministic embedding-input serialization."""

from fastapi_ctx_gateway.cache.serialize import canonicalize_turns
from fastapi_ctx_gateway.schemas.neutral import BinaryPart, TextPart, Turn


def test_deterministic_for_identical_turns() -> None:
    turns = [Turn(role="user", parts=[TextPart(text="hi")])]
    assert canonicalize_turns(turns) == canonicalize_turns(turns)


def test_differs_when_text_differs() -> None:
    a = [Turn(role="user", parts=[TextPart(text="hi")])]
    b = [Turn(role="user", parts=[TextPart(text="bye")])]
    assert canonicalize_turns(a) != canonicalize_turns(b)


def test_is_order_sensitive() -> None:
    a = [
        Turn(role="user", parts=[TextPart(text="first")]),
        Turn(role="assistant", parts=[TextPart(text="second")]),
    ]
    b = [
        Turn(role="assistant", parts=[TextPart(text="second")]),
        Turn(role="user", parts=[TextPart(text="first")]),
    ]
    assert canonicalize_turns(a) != canonicalize_turns(b)


def test_ignores_non_text_parts() -> None:
    with_binary = [
        Turn(
            role="user",
            parts=[TextPart(text="hi"), BinaryPart(mime_type="image/png", data="AAAA")],
        )
    ]
    without_binary = [Turn(role="user", parts=[TextPart(text="hi")])]
    assert canonicalize_turns(with_binary) == canonicalize_turns(without_binary)
