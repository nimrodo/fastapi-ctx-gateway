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
