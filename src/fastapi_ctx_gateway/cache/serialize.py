"""Deterministic serialization of pruned turns into embedding input."""

from fastapi_ctx_gateway.schemas.neutral import TextPart, Turn

__all__ = ["canonicalize_turns"]


def canonicalize_turns(turns: list[Turn]) -> str:
    """Serialize role+text turns, in order, ignoring non-text parts.

    Order-sensitive and text-only by design: what gets embedded/cached
    must reflect the same scope pruning already reduced content to, and
    must never depend on binary content the vectorizer can't reason about.
    """
    serialized = []
    for turn in turns:
        texts = [part.text for part in turn.parts if isinstance(part, TextPart)]
        serialized.append(turn.role + "\x00" + "\x00".join(texts))
    return "\x1e".join(serialized)
