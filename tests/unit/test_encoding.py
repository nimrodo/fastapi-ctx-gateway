"""Tests for encode_fields: the shared injective encoding helper."""

from fastapi_ctx_gateway.encoding import encode_fields


def test_deterministic_for_identical_input() -> None:
    assert encode_fields(["a", "b"]) == encode_fields(["a", "b"])


def test_no_intra_field_collision_from_embedded_separator_bytes() -> None:
    # The old \x00-join scheme collided here: "user\x00A\x00B" was produced
    # by both ["A", "B"] and ["A\x00B"].
    assert encode_fields(["A", "B"]) != encode_fields(["A\x00B"])


def test_no_cross_field_collision_from_embedded_colon_or_digits() -> None:
    # A field that looks like another field's length prefix must not be
    # able to shift the parse.
    assert encode_fields(["1", "a"]) != encode_fields(["1:a"])


def test_empty_field_list_is_distinct_from_a_single_empty_field() -> None:
    assert encode_fields([]) != encode_fields([""])
