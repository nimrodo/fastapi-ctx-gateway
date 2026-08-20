# Adding a provider

The gateway speaks its own neutral request/response/error contract; a [`Provider`][fastapi_ctx_gateway.providers.base.Provider] translates that contract to/from one upstream LLM API. `GeminiProvider` and `OpenAIProvider` are the two existing implementations — read both before writing a third; they're the worked examples this page assumes you have open.

See [ADR-0006](../adr/0006-neutral-schema-and-provider-abstraction.md) for why this abstraction exists at all.

## The interface

```python
class Provider(ABC):
    name: str

    @abstractmethod
    def stream(self, model: str, request: NeutralGenerateRequest) -> AsyncIterator[bytes]: ...
```

- `name` is the path segment clients use to reach this provider: `POST /v1/{name}/{model}:streamGenerateContent`.
- `stream()` is the entire contract. Given a neutral request, call the upstream API and yield **neutral** SSE bytes — one upstream event translates to exactly one yielded chunk, never re-buffered or re-split across more than one upstream event's worth of latency.
- **`stream()` must never raise** for an upstream or transport failure. Translate those in-band to a terminal `NeutralErrorEvent` chunk instead (see [`neutral_error_event`][fastapi_ctx_gateway.providers.sse.neutral_error_event]) — the router assumes every call to `stream()` yields a well-formed stream, error or not, and never wraps it in its own `try`/`except`.

## Steps

1. **Create `src/fastapi_ctx_gateway/providers/<name>.py`** implementing `Provider`. If the upstream API has its own wire schema worth modeling as Pydantic types (Gemini does, via `providers/gemini_wire.py`), keep those types in this module or an adapter-private sibling — they are never the gateway's own contract, only this adapter's internal detail.

2. **Request translation: neutral → native.** Walk `request.turns` (role + `parts`, each either `TextPart` or `BinaryPart`) and `request.system`, mapping to whatever shape the upstream API expects. `request.generation_config` maps field-by-field (`temperature`, `top_p`, `max_output_tokens`, `stop_sequences`, `candidate_count` — drop silently whatever the upstream doesn't support, as `OpenAIProvider` does for `top_k`). `request.tools`/`request.tool_config` are intentionally opaque `dict[str, Any]` pass-through in the neutral schema — real per-provider tool-calling translation is deferred until a provider actually needs it (see ADR-0006's consequences).

3. **Response translation: native SSE → neutral SSE, one event → one event.** Use the shared framing helper [`iter_sse_data_lines`][fastapi_ctx_gateway.providers.sse.iter_sse_data_lines] — it buffers only up to the next SSE event boundary (never the whole response) and yields each event's `data:` payload. Parse that payload into a `NeutralStreamEvent` (`delta.parts`, `finish_reason`, `usage`) and re-encode it as one `data: {...}\n\n` chunk. If the upstream has its own stream-terminator sentinel that isn't real content (OpenAI's literal `data: [DONE]`), swallow it rather than translating it into an empty neutral event.

4. **Errors.** On a non-2xx response or a transport exception, yield [`neutral_error_event`][fastapi_ctx_gateway.providers.sse.neutral_error_event]. Use [`parse_error_message`][fastapi_ctx_gateway.providers.sse.parse_error_message] to extract a readable message from a JSON error body (most providers, including Gemini's and OpenAI's, use `{"error": {"message": ...}}`) rather than dumping the raw bytes at the client.

5. **Retry policy.** Both existing providers implement one bounded pre-stream retry — a failure before any bytes reached the client retries once; a failure after bytes have streamed never retries. This lives *inside* the provider (see `GeminiProvider._MAX_RETRIES`), not in the generic router-level `stream_with_retry`, precisely because `stream()` never raises — the router's own retry-on-exception path is only a safety net for bugs the adapter itself doesn't catch.

6. **Config.** Add settings for this provider's credentials/base URL to `Settings` (`config.py`), following the `openai_*` fields as the pattern: **optional as a group** (`<name>_api_key: SecretStr | None = None`), never required like Gemini's — a deployment that doesn't use this provider shouldn't be forced to configure it. Treat an empty string the same as unset (see `_openai_is_configured` in `app.py`) rather than registering a provider that can never authenticate.

7. **Register it in `app.py`.** Add the provider to `_registered_provider_names()` (gated on its credentials being configured) and construct it in `_build_providers()`. `_registered_provider_names()` is also what drives the per-provider circuit-breaker registry (`app.state.circuit_breakers`) — get this step right and the breaker isolation, the `/v1/{name}/...` 404-when-unconfigured behavior, and the `circuit_breaker_open_total{provider}` metric label all follow automatically; nothing else in the pipeline (pruner, rate limiter, cache) needs to know this provider exists.

## Testing

Mirror the existing providers' test layout:

- `tests/unit/providers/test_<name>_request_translation.py` — unit tests for the neutral→native request builder, one test per field/branch (roles, system, binary parts, generation config, tools pass-through).
- `tests/unit/providers/test_<name>_response_translation.py` — unit tests for native SSE → neutral SSE, using `respx` to mock the upstream HTTP call; cover the 1:1 event-boundary guarantee, a non-2xx error, and a transport failure (`httpx.ConnectError` or similar).
- `tests/integration/test_stream_endpoint_<name>.py` — one router-level test proving the whole path works through `create_app()` and the real `/v1/{name}/{model}:streamGenerateContent` route, respx-mocking only the upstream HTTP call.
- Extend `tests/unit/test_app.py` with a provider-registered/not-registered pair, matching `test_openai_provider_registered_when_key_set`/`test_openai_provider_not_registered_when_key_unset`.

No test in this repo hits a real upstream API anywhere — keep it that way; mocked (`respx`) coverage is the standard here, not a real smoke test against the live provider.
