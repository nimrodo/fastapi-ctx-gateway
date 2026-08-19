"""Shared helpers for building Gemini-native mock SSE and expected neutral SSE bytes.

Used by integration tests that respx-mock Gemini's own wire format upstream
but assert on the gateway's translated neutral SSE bytes downstream.
"""

import json

__all__ = ["gemini_sse_event", "neutral_sse_event"]


def gemini_sse_event(
    text: str | None = None,
    finish_reason: str | None = None,
    total_tokens: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> bytes:
    """Build one Gemini-native SSE `data:` event, as Gemini itself would send it."""
    candidate: dict = {}
    if text is not None:
        candidate["content"] = {"parts": [{"text": text}], "role": "model"}
    if finish_reason is not None:
        candidate["finishReason"] = finish_reason
    payload: dict = {"candidates": [candidate]} if candidate else {"candidates": []}
    usage = {}
    if total_tokens is not None:
        usage["totalTokenCount"] = total_tokens
    if prompt_tokens is not None:
        usage["promptTokenCount"] = prompt_tokens
    if completion_tokens is not None:
        usage["candidatesTokenCount"] = completion_tokens
    if usage:
        payload["usageMetadata"] = usage
    return f"data: {json.dumps(payload)}\n\n".encode()


def neutral_sse_event(
    text: str | None = None,
    finish_reason: str | None = None,
    total_tokens: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> bytes:
    """Build the neutral SSE `data:` event the gateway is expected to translate to."""
    payload: dict = {}
    if text is not None:
        payload["delta"] = {"role": "assistant", "parts": [{"type": "text", "text": text}]}
    if finish_reason is not None:
        payload["finish_reason"] = finish_reason
    usage = {}
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if usage:
        payload["usage"] = usage
    # Compact separators to match pydantic's model_dump_json() byte-for-byte.
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
