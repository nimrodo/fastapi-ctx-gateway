"""Deterministic serialization of pruned contents into embedding input."""

from fastapi_ctx_gateway.schemas.gemini import Content

__all__ = ["canonicalize_contents"]


def canonicalize_contents(contents: list[Content]) -> str:
    """Serialize role+text turns, in order, ignoring non-text parts.

    Order-sensitive and text-only by design: what gets embedded/cached
    must reflect the same scope pruning already reduced content to, and
    must never depend on binary content the vectorizer can't reason about.
    """
    turns = []
    for content in contents:
        texts = [part.text for part in content.parts if part.text is not None]
        turns.append(content.role + "\x00" + "\x00".join(texts))
    return "\x1e".join(turns)
