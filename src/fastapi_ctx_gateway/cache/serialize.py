"""Deterministic serialization of pruned turns into embedding input."""

from fastapi_ctx_gateway.encoding import encode_fields
from fastapi_ctx_gateway.schemas.neutral import TextPart, Turn

__all__ = ["canonicalize_turns"]


def canonicalize_turns(turns: list[Turn]) -> str:
    """Serialize role+text turns, in order, ignoring non-text parts.

    Order-sensitive and text-only by design: what gets embedded/cached
    must reflect the same scope pruning already reduced content to, and
    must never depend on binary content the vectorizer can't reason about.

    Each turn's (role, *texts) is encoded with `encode_fields` (collision-
    free per turn), and the list of per-turn encodings is itself encoded
    with `encode_fields` again — nesting the same injective encoding one
    layer deeper so attacker-controlled text can't forge a fake turn
    boundary, not just a fake field boundary within one turn.
    """
    per_turn = []
    for turn in turns:
        texts = [part.text for part in turn.parts if isinstance(part, TextPart)]
        per_turn.append(encode_fields([turn.role, *texts]))
    return encode_fields(per_turn)
