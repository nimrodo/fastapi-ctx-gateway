"""Tests for OpenAIProvider's neutral -> OpenAI-native Chat Completions request translation."""

from fastapi_ctx_gateway.providers.openai import _to_openai_request
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    GenerationConfig,
    NeutralGenerateRequest,
    TextPart,
    Turn,
)


def test_translates_turns_and_keeps_role_names_as_is() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(role="user", parts=[TextPart(text="hi")]),
            Turn(role="assistant", parts=[TextPart(text="hello")]),
        ]
    )
    body = _to_openai_request("gpt-4o", request)
    assert body["model"] == "gpt-4o"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant"]
    assert body["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


def test_translates_system_to_a_leading_system_message() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        system=[TextPart(text="be nice")],
    )
    body = _to_openai_request("gpt-4o", request)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == [{"type": "text", "text": "be nice"}]
    assert body["messages"][1]["role"] == "user"


def test_no_system_message_when_system_is_unset() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    body = _to_openai_request("gpt-4o", request)
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_translates_binary_part_with_inline_data_to_a_data_uri() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="AAAA")])]
    )
    body = _to_openai_request("gpt-4o", request)
    part = body["messages"][0]["content"][0]
    assert part == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_translates_binary_part_with_uri_directly() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", uri="https://x/y.png")])]
    )
    body = _to_openai_request("gpt-4o", request)
    part = body["messages"][0]["content"][0]
    assert part == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}


def test_translates_generation_config() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        generation_config=GenerationConfig(
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=100,
            stop_sequences=["END"],
            candidate_count=2,
        ),
    )
    body = _to_openai_request("gpt-4o", request)
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 100
    assert body["stop"] == ["END"]
    assert body["n"] == 2
    # top_k has no OpenAI equivalent and is silently dropped.
    assert "top_k" not in body


def test_tools_pass_through_opaque() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        tool_config={"type": "auto"},
    )
    body = _to_openai_request("gpt-4o", request)
    assert body["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]
    assert body["tool_choice"] == {"type": "auto"}
