"""Tests for GeminiProvider's neutral -> Gemini-native request translation."""

from fastapi_ctx_gateway.providers.gemini import _to_gemini_request
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    GenerationConfig,
    NeutralGenerateRequest,
    TextPart,
    Turn,
)


def test_translates_turns_and_maps_assistant_role_to_model() -> None:
    request = NeutralGenerateRequest(
        turns=[
            Turn(role="user", parts=[TextPart(text="hi")]),
            Turn(role="assistant", parts=[TextPart(text="hello")]),
        ]
    )
    gemini_request = _to_gemini_request(request)
    assert [c.role for c in gemini_request.contents] == ["user", "model"]
    assert gemini_request.contents[0].parts[0].text == "hi"


def test_translates_system_to_system_instruction() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        system=[TextPart(text="be nice")],
    )
    gemini_request = _to_gemini_request(request)
    assert gemini_request.system_instruction is not None
    assert gemini_request.system_instruction.parts[0].text == "be nice"


def test_no_system_instruction_when_system_is_unset() -> None:
    request = NeutralGenerateRequest(turns=[Turn(role="user", parts=[TextPart(text="hi")])])
    gemini_request = _to_gemini_request(request)
    assert gemini_request.system_instruction is None


def test_translates_binary_part_with_inline_data() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", data="AAAA")])]
    )
    gemini_request = _to_gemini_request(request)
    part = gemini_request.contents[0].parts[0]
    assert part.inline_data == {"mimeType": "image/png", "data": "AAAA"}
    assert part.file_data is None


def test_translates_binary_part_with_uri_as_file_data() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[BinaryPart(mime_type="image/png", uri="gs://bucket/f")])]
    )
    gemini_request = _to_gemini_request(request)
    part = gemini_request.contents[0].parts[0]
    assert part.file_data == {"mimeType": "image/png", "fileUri": "gs://bucket/f"}
    assert part.inline_data is None


def test_translates_generation_config_and_tools_passthrough() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        generation_config=GenerationConfig(temperature=0.5, max_output_tokens=100),
        tools=[{"functionDeclarations": []}],
        tool_config={"functionCallingConfig": {"mode": "AUTO"}},
        safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT"}],
    )
    gemini_request = _to_gemini_request(request)
    assert gemini_request.generation_config is not None
    assert gemini_request.generation_config.temperature == 0.5
    assert gemini_request.generation_config.max_output_tokens == 100
    assert gemini_request.tools == [{"functionDeclarations": []}]
    assert gemini_request.tool_config == {"functionCallingConfig": {"mode": "AUTO"}}
    assert gemini_request.safety_settings == [{"category": "HARM_CATEGORY_HARASSMENT"}]
