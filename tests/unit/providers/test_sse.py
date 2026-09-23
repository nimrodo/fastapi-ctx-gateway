"""Tests for shared SSE/error helpers used by every provider adapter."""

import json

from fastapi_ctx_gateway.providers.sse import parse_error_message


def test_extracts_message_from_json_error_envelope() -> None:
    body = json.dumps({"error": {"message": "invalid api key", "type": "invalid_request"}}).encode()
    result = parse_error_message("OpenAI", 401, body)
    assert result == "OpenAI returned 401: invalid api key"


def test_falls_back_to_raw_text_when_body_is_not_json() -> None:
    result = parse_error_message("OpenAI", 500, b"internal server error")
    assert result == "OpenAI returned 500: internal server error"


def test_falls_back_to_raw_text_when_json_has_no_error_message() -> None:
    body = json.dumps({"detail": "something else"}).encode()
    result = parse_error_message("OpenAI", 400, body)
    assert result == 'OpenAI returned 400: {"detail": "something else"}'


def test_omits_body_entirely_when_empty() -> None:
    result = parse_error_message("OpenAI", 503, b"")
    assert result == "OpenAI returned 503"


def test_falls_back_to_raw_text_when_error_message_is_not_a_string() -> None:
    body = json.dumps({"error": {"message": {"nested": "object"}}}).encode()
    result = parse_error_message("Gemini", 400, body)
    assert result == 'Gemini returned 400: {"error": {"message": {"nested": "object"}}}'


def test_redacts_a_passed_in_secret_from_the_relayed_message() -> None:
    body = json.dumps(
        {"error": {"message": "Incorrect API key provided: sk-mysecretgatewaykey123."}}
    ).encode()
    result = parse_error_message("OpenAI", 401, body, secrets=["sk-mysecretgatewaykey123"])
    assert "sk-mysecretgatewaykey123" not in result
    assert result == "OpenAI returned 401: Incorrect API key provided: [REDACTED]."


def test_redacts_key_shaped_fragments_even_when_not_passed_as_a_secret() -> None:
    body = json.dumps({"error": {"message": "bad key sk-abcdefghijklmnop"}}).encode()
    result = parse_error_message("OpenAI", 401, body)
    assert "sk-abcdefghijklmnop" not in result
    assert "[REDACTED]" in result


def test_does_not_redact_a_secret_shorter_than_the_minimum_length() -> None:
    """A too-short "secret" (e.g. a 1-char test/placeholder key) is left
    alone rather than blindly substring-replaced, which would otherwise
    mangle ordinary words that happen to contain it (see redact_secrets's
    _MIN_REDACTABLE_SECRET_LENGTH docstring) — a real upstream key is
    never this short, so nothing meaningful goes unredacted in practice.
    """
    body = json.dumps({"error": {"message": "API key not valid"}}).encode()
    result = parse_error_message("Gemini", 400, body, secrets=["k"])
    assert result == "Gemini returned 400: API key not valid"


def test_regex_backstop_also_catches_anthropic_and_gemini_key_shapes() -> None:
    anthropic_body = json.dumps({"error": {"message": "bad key sk-ant-abcdefghijklmnop"}}).encode()
    gemini_body = json.dumps({"error": {"message": "bad key AIzaSyAbcdefghijklmnop"}}).encode()
    assert "sk-ant-" not in parse_error_message("Anthropic", 401, anthropic_body)
    assert "AIza" not in parse_error_message("Gemini", 400, gemini_body)
