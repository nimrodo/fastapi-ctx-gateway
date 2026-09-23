"""Anthropic provider adapter: neutral contract <-> the Claude Python SDK.

Unlike `GeminiProvider`/`OpenAIProvider`, this adapter is built on the official
`anthropic` SDK rather than raw HTTP. Consequences (see ADR-0007):

- streaming yields typed SDK events, not `data:` byte frames, so
  `providers/sse.py`'s framing helper is unused (only `neutral_error_event` is);
- the SDK raises typed exceptions, so `stream()` wraps every call to honour the
  never-raise contract in `providers/base.py`;
- retry is the SDK's own (`max_retries=1` on the injected client), not a manual
  pre-stream loop — for a streaming call that retry is entirely pre-first-event,
  which is exactly the "retry only before bytes reached the client" contract;
- the SDK client owns an httpx2 connection pool; `aclose()` closes it and the
  app's lifespan calls that on shutdown.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import anthropic

from fastapi_ctx_gateway.providers.base import Provider
from fastapi_ctx_gateway.providers.sse import neutral_error_event, redact_secrets
from fastapi_ctx_gateway.schemas.neutral import (
    BinaryPart,
    Delta,
    FinishReason,
    NeutralGenerateRequest,
    NeutralStreamEvent,
    TextPart,
    Turn,
    Usage,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

__all__ = ["AnthropicProvider"]

_FINISH_REASON_FROM_ANTHROPIC = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.MAX_TOKENS,
    "refusal": FinishReason.SAFETY,
    "tool_use": FinishReason.OTHER,
    "pause_turn": FinishReason.OTHER,
}


class AnthropicProvider(Provider):
    """Translates the neutral contract to/from Anthropic's Messages streaming API."""

    name = "anthropic"

    def __init__(
        self, client: "AsyncAnthropic", default_max_tokens: int, api_key: str = ""
    ) -> None:
        """Wrap an injected SDK client.

        Tests pass a client built over a stubbed transport here; production
        goes through `from_settings`. `default_max_tokens` is the fallback for
        Anthropic's mandatory `max_tokens` when a neutral request omits
        `generation_config.max_output_tokens`. `api_key` (default "" for
        tests that don't care) is redacted from any relayed error message —
        see `_error_message`.
        """
        self._client = client
        self._default_max_tokens = default_max_tokens
        self._api_key = api_key

    @classmethod
    def from_settings(
        cls,
        *,
        api_key: str,
        base_url: str | None,
        default_max_tokens: int,
        max_retries: int = 1,
        timeout: float = 30.0,
    ) -> "AnthropicProvider":
        """Build a provider with a real SDK client. Called by the provider registry.

        `max_retries=1` stands in for the bounded pre-stream retry the
        raw-HTTP adapters implement by hand — for a streaming call the SDK's
        retry is entirely pre-first-event.
        """
        client = anthropic.AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
            timeout=timeout,
        )
        return cls(client=client, default_max_tokens=default_max_tokens, api_key=api_key)

    async def aclose(self) -> None:
        """Close the SDK client's connection pool. Called by the app's lifespan."""
        await self._client.close()

    async def stream(self, model: str, request: NeutralGenerateRequest) -> AsyncIterator[bytes]:
        """Call Anthropic via the SDK and yield neutral SSE bytes, one native event per chunk."""
        kwargs = _to_anthropic_request(model, request, self._default_max_tokens)
        try:
            events = await self._client.messages.create(model=model, stream=True, **kwargs)
        except anthropic.APIStatusError as exc:
            yield neutral_error_event(_error_message(exc, self._api_key), exc.status_code)
            return
        except (anthropic.APIConnectionError, anthropic.APIError) as exc:
            yield neutral_error_event(str(exc), None)
            return

        try:
            async for chunk in _translate_events(events):
                yield chunk
        except (anthropic.APIConnectionError, anthropic.APIError) as exc:
            # A mid-stream transport drop: bytes may already have reached the
            # client, so this is never retried — just emit a terminal error.
            yield neutral_error_event(str(exc), None)
        finally:
            await events.close()


def _error_message(exc: anthropic.APIStatusError, api_key: str) -> str:
    """Build the relayed error message, redacted like every other provider's.

    Anthropic's documented error bodies carry no content or key fragments
    today (see docs/security.md), but this participates in the same
    redaction interface as Gemini/OpenAI anyway so the key-shape regex
    backstop still applies here, and so that fact staying true isn't a
    structural guarantee this code silently depends on.
    """
    body = exc.body
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            message = f"Anthropic returned {exc.status_code}: {err['message']}"
            return redact_secrets(message, secrets=[api_key])
    message = f"Anthropic returned {exc.status_code}: {exc.message}"
    return redact_secrets(message, secrets=[api_key])


# --- request translation: neutral -> anthropic SDK kwargs ---


def _part_to_anthropic_content(part: TextPart | BinaryPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    source = (
        {"type": "base64", "media_type": part.mime_type, "data": part.data}
        if part.data is not None
        else {"type": "url", "url": part.uri}
    )
    return {"type": "image", "source": source}


def _turn_to_anthropic_message(turn: Turn) -> dict[str, Any]:
    return {
        "role": turn.role,
        "content": [_part_to_anthropic_content(p) for p in turn.parts],
    }


def _to_anthropic_request(
    model: str, request: NeutralGenerateRequest, default_max_tokens: int
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [_turn_to_anthropic_message(turn) for turn in request.turns],
        "max_tokens": default_max_tokens,
    }

    if request.system:
        # Anthropic's `system` takes text only — a BinaryPart there is dropped.
        text_blocks = [
            {"type": "text", "text": p.text} for p in request.system if isinstance(p, TextPart)
        ]
        if text_blocks:
            body["system"] = text_blocks

    if request.tools:
        body["tools"] = request.tools
    if request.tool_config:
        body["tool_choice"] = request.tool_config
    # safety_settings has no Anthropic equivalent and is intentionally dropped.

    config = request.generation_config
    if config is not None:
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.top_p is not None:
            body["top_p"] = config.top_p
        if config.top_k is not None:
            # Kept — Anthropic supports top_k (OpenAI/Gemini adapters drop it).
            body["top_k"] = config.top_k
        if config.max_output_tokens is not None:
            body["max_tokens"] = config.max_output_tokens
        if config.stop_sequences is not None:
            body["stop_sequences"] = config.stop_sequences
        # candidate_count has no Anthropic equivalent (no n>1) and is dropped.
    return body


# --- response translation: anthropic SDK events -> neutral SSE, one event -> 0 or 1 ---


async def _translate_events(events: AsyncIterator[Any]) -> AsyncIterator[bytes]:
    """Translate each SDK stream event to at most one neutral SSE event.

    Anthropic splits token accounting across events: `message_start` carries
    the prompt tokens, `message_delta` carries the completion tokens and the
    stop reason. We hold the prompt-token count and emit one fully-populated
    `Usage` (prompt/completion/total) on the `message_delta` event, computing
    `total_tokens` ourselves — the SDK never sends it, and the router
    reconciles rate-limit against it.
    """
    input_tokens = 0
    async for event in events:
        event_type = getattr(event, "type", None)

        if event_type == "message_start":
            usage = event.message.usage
            input_tokens = (
                (usage.input_tokens or 0)
                + (getattr(usage, "cache_read_input_tokens", 0) or 0)
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
            )
            continue

        if event_type == "content_block_delta":
            delta = event.delta
            if getattr(delta, "type", None) == "text_delta" and delta.text:
                neutral = NeutralStreamEvent(
                    delta=Delta(role="assistant", parts=[TextPart(text=delta.text)])
                )
                yield _encode(neutral)
            # thinking_delta / signature_delta / input_json_delta carry no
            # neutral text content and are swallowed.
            continue

        if event_type == "message_delta":
            native_reason = getattr(event.delta, "stop_reason", None)
            finish_reason = (
                _FINISH_REASON_FROM_ANTHROPIC.get(native_reason, FinishReason.OTHER)
                if native_reason
                else None
            )
            delta_usage = getattr(event, "usage", None)
            usage = None
            if delta_usage is not None:
                output_tokens = getattr(delta_usage, "output_tokens", None) or 0
                prompt_tokens = getattr(delta_usage, "input_tokens", None) or input_tokens
                usage = Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=output_tokens,
                    total_tokens=prompt_tokens + output_tokens,
                )
            yield _encode(NeutralStreamEvent(finish_reason=finish_reason, usage=usage))
            continue

        # message_stop / content_block_start / content_block_stop / ping:
        # no neutral content, swallowed (cf. OpenAI's [DONE]).


def _encode(event: NeutralStreamEvent) -> bytes:
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n".encode()
