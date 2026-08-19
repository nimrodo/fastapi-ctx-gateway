"""The gateway's own neutral request/response/error contract.

Provider-agnostic by construction: nothing here is modeled on Gemini's or
OpenAI's wire schema. Provider adapters (see fastapi_ctx_gateway.providers)
translate to/from this shape; nothing above the provider boundary should
ever see a provider-native type.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BinaryPart",
    "Delta",
    "FinishReason",
    "GenerationConfig",
    "NeutralError",
    "NeutralErrorEvent",
    "NeutralGenerateRequest",
    "NeutralStreamEvent",
    "Part",
    "TextPart",
    "Turn",
    "Usage",
]


class _NeutralModel(BaseModel):
    """Base for the gateway's own contract: strict, not a validation-only mirror."""

    model_config = ConfigDict(extra="forbid")


class TextPart(_NeutralModel):
    """A prunable/embeddable piece of text within a turn."""

    type: Literal["text"] = "text"
    text: str


class BinaryPart(_NeutralModel):
    """Inline (base64) or by-reference (uri) non-text content."""

    type: Literal["binary"] = "binary"
    mime_type: str
    data: str | None = None
    uri: str | None = None


Part = Annotated[TextPart | BinaryPart, Field(discriminator="type")]


class Turn(_NeutralModel):
    """One turn of a conversation: a role and its parts."""

    role: Literal["user", "assistant"]
    parts: list[Part] = []


class GenerationConfig(_NeutralModel):
    """Sampling/output configuration for a generate request."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    candidate_count: int | None = None


class NeutralGenerateRequest(_NeutralModel):
    """Body of a streaming generate request, provider-agnostic."""

    turns: list[Turn]
    system: list[Part] | None = None
    generation_config: GenerationConfig | None = None
    tools: list[dict[str, Any]] | None = None
    tool_config: dict[str, Any] | None = None
    safety_settings: list[dict[str, Any]] | None = None


class Delta(_NeutralModel):
    """The incremental content of one streamed event."""

    role: Literal["assistant"] | None = None
    parts: list[Part] = []


class FinishReason(StrEnum):
    """Why a stream stopped, normalized across providers."""

    STOP = "stop"
    MAX_TOKENS = "max_tokens"
    SAFETY = "safety"
    OTHER = "other"


class Usage(_NeutralModel):
    """Token accounting for a completed request, normalized across providers."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class NeutralStreamEvent(_NeutralModel):
    """One SSE event on the gateway's streamed response."""

    delta: Delta | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class NeutralError(_NeutralModel):
    """A provider-agnostic error, normalized from whatever the upstream returned."""

    message: str
    type: str
    provider_status: int | None = None


class NeutralErrorEvent(_NeutralModel):
    """A terminal error SSE event."""

    error: NeutralError
