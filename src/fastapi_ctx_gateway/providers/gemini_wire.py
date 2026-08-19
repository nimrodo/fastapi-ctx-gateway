"""Gemini's own wire schema.

Field names mirror Gemini's REST JSON casing (camelCase) via an alias
generator; Python code uses snake_case attributes. This is used only inside
the Gemini provider adapter (`fastapi_ctx_gateway.providers.gemini`) for
request/response translation — it is not the gateway's public contract; see
`fastapi_ctx_gateway.schemas.neutral` for that.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

__all__ = [
    "Content",
    "GenerateContentRequest",
    "GenerationConfig",
    "Part",
]


class _GeminiModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")


class Part(_GeminiModel):
    """A single piece of a turn: text, or a non-text (image/audio/file) blob."""

    text: str | None = None
    inline_data: dict[str, Any] | None = None
    file_data: dict[str, Any] | None = None

    @property
    def is_text(self) -> bool:
        """Whether this part is prunable/embeddable text (vs. binary content)."""
        return self.text is not None


class Content(_GeminiModel):
    """One turn: a role ("user"/"model") and its parts."""

    role: str
    parts: list[Part] = []


class GenerationConfig(_GeminiModel):
    """Sampling/output configuration for a generate request."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None
    candidate_count: int | None = None


class GenerateContentRequest(_GeminiModel):
    """Body of a (stream)GenerateContent call. `model` comes from the URL, not this body."""

    contents: list[Content]
    tools: list[dict[str, Any]] | None = None
    tool_config: dict[str, Any] | None = None
    safety_settings: list[dict[str, Any]] | None = None
    system_instruction: Content | None = None
    generation_config: GenerationConfig | None = None
