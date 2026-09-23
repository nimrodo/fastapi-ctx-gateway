"""Shared low-level SSE framing and error-event helpers, used by every provider adapter.

Each provider's own response translator still owns interpreting the JSON
payload of an event — only the "how do we split raw upstream bytes into
per-event `data:` payloads, buffering at most one event at a time" and "how
do we build the terminal neutral error event" pieces are shared, since both
are identical regardless of which upstream API is being called.
"""

import json
import re
from collections.abc import AsyncIterator, Iterable

from fastapi_ctx_gateway.schemas.neutral import NeutralError, NeutralErrorEvent

__all__ = ["iter_sse_data_lines", "neutral_error_event", "parse_error_message", "redact_secrets"]

_REDACTED = "[REDACTED]"

# Provider-agnostic backstop for common upstream key shapes (OpenAI's
# `sk-...`, Anthropic's `sk-ant-...`, Google's `AIza...`). Applied
# independently of any explicitly-passed secret, so a leak doesn't solely
# depend on the caller remembering to pass every credential that could
# appear in a relayed error body.
_KEY_SHAPE_PATTERN = re.compile(r"(?:sk-ant-|sk-|AIza)[A-Za-z0-9_-]{10,}")

# A real upstream API key is never this short; below this length, blind
# substring replacement risks mangling ordinary words that happen to
# contain the "secret" (e.g. a single-character test/placeholder key)
# rather than redacting anything meaningful.
_MIN_REDACTABLE_SECRET_LENGTH = 8


def redact_secrets(message: str, secrets: Iterable[str] = ()) -> str:
    """Scrub known secrets and common key-shaped substrings from a message.

    Each literal occurrence of a passed-in secret is replaced first, then
    the provider-agnostic key-shape regex backstop runs over what's left —
    so a credential leak isn't solely dependent on the caller passing the
    exact right secret ahead of time.
    """
    for secret in secrets:
        if secret and len(secret) >= _MIN_REDACTABLE_SECRET_LENGTH:
            message = message.replace(secret, _REDACTED)
    return _KEY_SHAPE_PATTERN.sub(_REDACTED, message)


async def iter_sse_data_lines(native_bytes: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    r"""Buffer only up to the next SSE event boundary and yield each event's `data:` payload.

    Never buffers more than one event's worth of bytes at a time — this is
    what keeps upstream-event -> neutral-event translation 1:1 without
    adding more than one event's worth of latency. A trailing partial
    buffer (no closing `\n\n`) has nothing to yield.
    """
    buffer = b""
    async for chunk in native_bytes:
        buffer += chunk
        while b"\n\n" in buffer:
            raw_event, buffer = buffer.split(b"\n\n", 1)
            for line in raw_event.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[len(b"data:") :].strip()
                if payload:
                    yield payload
                break


def neutral_error_event(
    message: str, status: int | None, error_type: str = "upstream_error"
) -> bytes:
    """Build the terminal neutral SSE error event every provider yields on failure."""
    payload = NeutralErrorEvent(
        error=NeutralError(message=message, type=error_type, provider_status=status)
    )
    return f"data: {payload.model_dump_json()}\n\n".encode()


def parse_error_message(
    provider_label: str, status_code: int, body: bytes, secrets: Iterable[str] = ()
) -> str:
    """Build a human-readable error message from a non-2xx response body.

    Most providers' error envelopes are JSON-shaped (`{"error": {"message":
    ...}}` — both Gemini's and OpenAI's do this), so this extracts just the
    message rather than dumping the raw JSON blob at the client. Falls back
    to the raw decoded text if the body isn't JSON, doesn't have that
    shape, or `message` isn't a string.

    `secrets` (typically the provider's own configured upstream key) are
    redacted from the returned message, alongside the provider-agnostic
    key-shape backstop — see `redact_secrets`. This closes the OpenAI
    error-relay credential leak documented in docs/security.md, where a
    misconfigured/rotated upstream key gets echoed back in the 401 body.
    """
    text = body.decode(errors="replace").strip()
    if not text:
        return f"{provider_label} returned {status_code}"
    message = text
    try:
        parsed = json.loads(text)
        candidate = parsed.get("error", {}).get("message") if isinstance(parsed, dict) else None
        if isinstance(candidate, str) and candidate:
            message = candidate
    except (ValueError, AttributeError):
        pass
    return redact_secrets(f"{provider_label} returned {status_code}: {message}", secrets)
