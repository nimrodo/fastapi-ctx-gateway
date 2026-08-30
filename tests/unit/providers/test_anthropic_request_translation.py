"""Tests for AnthropicProvider's neutral -> anthropic SDK kwargs translation."""

from fastapi_ctx_gateway.providers.anthropic import _to_anthropic_request
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    GenerationConfig,
    NeutralGenerateRequest,
    TextPart,
    Turn,
)

_DEFAULT_MAX_TOKENS = 4096


def _req(**kwargs) -> NeutralGenerateRequest:
    kwargs.setdefault("turns", [Turn(role="user", parts=[TextPart(text="hi")])])
    return NeutralGenerateRequest(**kwargs)


def test_maps_roles_and_text_parts_one_to_one() -> None:
    request = _req(
        turns=[
            Turn(role="user", parts=[TextPart(text="q")]),
            Turn(role="assistant", parts=[TextPart(text="a")]),
        ]
    )
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
    ]


def test_does_not_massage_role_order() -> None:
    request = _req(turns=[Turn(role="assistant", parts=[TextPart(text="first")])])
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["messages"][0]["role"] == "assistant"


def test_system_text_parts_become_top_level_text_blocks() -> None:
    request = _req(system=[TextPart(text="be brief"), TextPart(text="be kind")])
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["system"] == [
        {"type": "text", "text": "be brief"},
        {"type": "text", "text": "be kind"},
    ]


def test_binary_part_in_system_is_dropped() -> None:
    request = _req(
        system=[TextPart(text="hi"), BinaryPart(mime_type="image/png", data="Zm9v")],
    )
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["system"] == [{"type": "text", "text": "hi"}]


def test_system_with_only_binary_parts_sets_no_system_field() -> None:
    request = _req(system=[BinaryPart(mime_type="image/png", data="Zm9v")])
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert "system" not in body


def test_binary_part_with_data_becomes_base64_image() -> None:
    request = _req(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="Zm9v")])]
    )
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["messages"][0]["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "Zm9v"},
    }


def test_binary_part_with_uri_becomes_url_image() -> None:
    request = _req(
        turns=[
            Turn(
                role="user",
                parts=[BinaryPart(mime_type="image/png", uri="https://ex.com/x.png")],
            )
        ]
    )
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["messages"][0]["content"][0] == {
        "type": "image",
        "source": {"type": "url", "url": "https://ex.com/x.png"},
    }


def test_max_tokens_defaults_when_generation_config_omits_it() -> None:
    body = _to_anthropic_request("claude-opus-5", _req(), _DEFAULT_MAX_TOKENS)
    assert body["max_tokens"] == _DEFAULT_MAX_TOKENS


def test_max_output_tokens_overrides_the_default() -> None:
    request = _req(generation_config=GenerationConfig(max_output_tokens=100))
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["max_tokens"] == 100


def test_top_k_is_forwarded() -> None:
    request = _req(generation_config=GenerationConfig(top_k=40))
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["top_k"] == 40


def test_sampling_fields_map_field_by_field() -> None:
    request = _req(
        generation_config=GenerationConfig(temperature=0.5, top_p=0.9, stop_sequences=["STOP"])
    )
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    assert body["stop_sequences"] == ["STOP"]


def test_candidate_count_is_dropped() -> None:
    request = _req(generation_config=GenerationConfig(candidate_count=3))
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert "n" not in body
    assert "candidate_count" not in body


def test_tools_and_tool_config_pass_through_opaquely() -> None:
    tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]
    request = _req(tools=tools, tool_config={"type": "auto"})
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert body["tools"] == tools
    assert body["tool_choice"] == {"type": "auto"}


def test_safety_settings_are_dropped() -> None:
    request = _req(safety_settings=[{"category": "HARM", "threshold": "BLOCK_NONE"}])
    body = _to_anthropic_request("claude-opus-5", request, _DEFAULT_MAX_TOKENS)
    assert "safety_settings" not in body
