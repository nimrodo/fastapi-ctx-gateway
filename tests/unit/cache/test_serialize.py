"""Tests for canonicalize_contents: deterministic embedding-input serialization."""

from fastapi_ctx_gateway.cache.serialize import canonicalize_contents
from fastapi_ctx_gateway.schemas.gemini import Content, Part


def test_deterministic_for_identical_contents() -> None:
    contents = [Content(role="user", parts=[Part(text="hi")])]
    assert canonicalize_contents(contents) == canonicalize_contents(contents)


def test_differs_when_text_differs() -> None:
    a = [Content(role="user", parts=[Part(text="hi")])]
    b = [Content(role="user", parts=[Part(text="bye")])]
    assert canonicalize_contents(a) != canonicalize_contents(b)


def test_is_order_sensitive() -> None:
    a = [
        Content(role="user", parts=[Part(text="first")]),
        Content(role="model", parts=[Part(text="second")]),
    ]
    b = [
        Content(role="model", parts=[Part(text="second")]),
        Content(role="user", parts=[Part(text="first")]),
    ]
    assert canonicalize_contents(a) != canonicalize_contents(b)


def test_ignores_non_text_parts() -> None:
    with_image = [
        Content(
            role="user",
            parts=[Part(text="hi"), Part(inline_data={"mimeType": "image/png", "data": "AAAA"})],
        )
    ]
    without_image = [Content(role="user", parts=[Part(text="hi")])]
    assert canonicalize_contents(with_image) == canonicalize_contents(without_image)
