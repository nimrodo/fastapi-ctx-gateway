"""Tests for the gateway's own neutral request/response/error contract."""

import pytest
from pydantic import ValidationError

from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    Delta,
    FinishReason,
    IntermediateStep,
    NeutralError,
    NeutralErrorEvent,
    NeutralGenerateRequest,
    NeutralStreamEvent,
    TextPart,
    Turn,
    Usage,
)


def test_turn_parses_text_part_via_type_discriminator() -> None:
    turn = Turn.model_validate({"role": "user", "parts": [{"type": "text", "text": "hi"}]})
    assert isinstance(turn.parts[0], TextPart)
    assert turn.parts[0].text == "hi"


def test_turn_parses_binary_part_via_type_discriminator() -> None:
    turn = Turn.model_validate(
        {
            "role": "user",
            "parts": [{"type": "binary", "mime_type": "image/png", "data": "AAAA"}],
        }
    )
    assert isinstance(turn.parts[0], BinaryPart)
    assert turn.parts[0].mime_type == "image/png"


def test_unknown_part_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Turn.model_validate({"role": "user", "parts": [{"type": "audio", "text": "hi"}]})


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        NeutralGenerateRequest.model_validate(
            {"turns": [], "unexpected_field": "should not be allowed"}
        )


def test_generate_request_round_trip() -> None:
    request = NeutralGenerateRequest(
        turns=[Turn(role="user", parts=[TextPart(text="hi")])],
        system=[TextPart(text="be nice")],
    )
    dumped = request.model_dump(exclude_none=True)
    assert dumped["turns"][0]["parts"][0]["text"] == "hi"
    assert dumped["system"][0]["text"] == "be nice"
    restored = NeutralGenerateRequest.model_validate(dumped)
    assert restored == request


def test_stream_event_round_trip_with_finish_reason_and_usage() -> None:
    event = NeutralStreamEvent(
        delta=Delta(role="assistant", parts=[TextPart(text="hi")]),
        finish_reason=FinishReason.STOP,
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    dumped = event.model_dump_json(exclude_none=True)
    restored = NeutralStreamEvent.model_validate_json(dumped)
    assert restored == event


def test_stream_event_omits_intermediate_by_default() -> None:
    """Gemini/OpenAI/Anthropic providers never set `intermediate`; it must not appear."""
    event = NeutralStreamEvent(delta=Delta(role="assistant", parts=[TextPart(text="hi")]))
    dumped = event.model_dump(exclude_none=True)
    assert "intermediate" not in dumped


def test_stream_event_round_trip_with_intermediate() -> None:
    event = NeutralStreamEvent(
        intermediate=IntermediateStep(label="tool_call", data={"name": "search", "args": {}})
    )
    dumped = event.model_dump_json(exclude_none=True)
    restored = NeutralStreamEvent.model_validate_json(dumped)
    assert restored == event


def test_intermediate_step_label_is_optional() -> None:
    step = IntermediateStep(data="raw payload")
    assert step.label is None
    assert step.data == "raw payload"


def test_intermediate_step_data_can_be_arbitrary_json() -> None:
    step = IntermediateStep(label="thought", data={"nested": [1, 2, {"three": 3}]})
    dumped = step.model_dump()
    assert dumped == {"label": "thought", "data": {"nested": [1, 2, {"three": 3}]}}


def test_error_event_shape() -> None:
    event = NeutralErrorEvent(
        error=NeutralError(message="boom", type="upstream_error", provider_status=502)
    )
    dumped = event.model_dump(exclude_none=True)
    assert dumped == {
        "error": {"message": "boom", "type": "upstream_error", "provider_status": 502}
    }
