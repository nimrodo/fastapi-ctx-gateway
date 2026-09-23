r"""Injective string encoding, shared by every canonicalization/fingerprint seam.

Both `cache/serialize.py::canonicalize_turns` and
`pruning.py::TokenBudgetPruner._fingerprint` used to join fields with a
fixed separator byte (`\x00`/`\x1e`). That's collision-prone: nothing
stops attacker-controlled text from containing the separator itself,
letting two genuinely different inputs serialize identically (see
`docs/security.md` and `docs/research/issue-8-canonicalization-collision.md`).

`encode_fields` replaces both ad-hoc joins with one netstring-style,
length-prefixed encoding — injective over the full `str` domain by
construction, since no field's content can ever be mistaken for another
field's length prefix or boundary.
"""

__all__ = ["encode_fields"]


def encode_fields(fields: list[str]) -> str:
    """Injectively encode a list of strings as one string.

    Each field is encoded as `f"{len(field)}:{field}"` (netstring-style)
    and the results are concatenated with no separator — none is needed,
    since the length prefix alone makes each field's boundary unambiguous.
    Two different field lists can never produce the same encoded output,
    regardless of what characters (including digits or ':') the fields
    themselves contain.

    Nest this call (encode a list of already-encoded strings) to extend
    the same guarantee across another structural layer, e.g. per-turn
    fields nested inside a list of turns — see `canonicalize_turns`.
    """
    return "".join(f"{len(field)}:{field}" for field in fields)
