# Configuration

The gateway is configured entirely through environment variables, prefixed `GATEWAY_`, loaded via [`Settings`][fastapi_ctx_gateway.config.Settings] (a `pydantic-settings` model). A `.env` file in the working directory is picked up automatically.

Copy the repo's [`.env.example`](https://github.com/nimrodo/fastapi-ctx-gateway/blob/main/.env.example) to `.env` and fill in real values as a starting point — it lists every variable below with the required ones at the top. `.env` itself is gitignored, so it's safe to put real keys in it.

## Required

```bash
GATEWAY_TENANT_API_KEYS='{"<gateway-issued-key>": "<tenant-id>"}'
# ...plus at least one provider key (see below)
GATEWAY_GEMINI_UPSTREAM_KEY=<your-gemini-api-key>
```

`GATEWAY_TENANT_API_KEYS` has no default and the app refuses to boot without it — an empty mapping would mean no client can ever authenticate, so an unconfigured gateway fails fast at startup instead of shipping a service that 401s every request. See [Multi-tenant keys](../advanced/multi-tenant-keys.md). Everything else has a working default.

**At least one provider must be configured.** No provider is mandatory (see [ADR-0008](../adr/0008-providers-are-opt-in-extras.md)): each is an optional config group *and* an optional dependency extra. Set a provider's key and install its extra (`fastapi-ctx-gateway[gemini]` / `[openai]` / `[anthropic]`, or `[all]`) and it registers; leave the key unset and `POST /v1/<name>/...` 404s. `create_app()` refuses to boot if *zero* providers end up configured. A key set without its extra installed fails at boot with an actionable message.

## Nested settings

Fields like `token_budgets` accept JSON directly:

```bash
GATEWAY_TOKEN_BUDGETS='{"budgets": {"gemini-3.7-flash": 32000, "gemini-2.5-pro": 64000}, "default": 16000}'
```

## Full reference

| Variable | Default | What it controls |
|---|---|---|
| `GATEWAY_REDIS_URL` | `redis://localhost:6379` | Shared state store (rate limits, semantic cache) |
| `GATEWAY_GEMINI_UPSTREAM_KEY` | *(unset)* | Sent to Gemini as `x-goog-api-key`. Unset (or blank) means the `gemini` provider isn't registered — never a boot failure |
| `GATEWAY_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | |
| `GATEWAY_OPENAI_API_KEY` | *(unset)* | Sent to OpenAI as `Authorization: Bearer`. Unset (or blank) means the `openai` provider isn't registered |
| `GATEWAY_OPENAI_BASE_URL` | `https://api.openai.com/v1` | |
| `GATEWAY_OPENAI_INCLUDE_USAGE` | `true` | Whether requests ask for `stream_options.include_usage`. Disable for an OpenAI-compatible server that rejects the field — see ADR-0006 |
| `GATEWAY_ANTHROPIC_API_KEY` | *(unset)* | Enables the `anthropic` provider (needs the `[anthropic]` extra). Unset (or blank) means it isn't registered |
| `GATEWAY_ANTHROPIC_BASE_URL` | *(unset)* | Overrides the `anthropic` SDK's default base URL (proxy / compatible gateway) |
| `GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS` | `4096` | Fallback for Anthropic's mandatory `max_tokens` when a request omits `generation_config.max_output_tokens` |
| `GATEWAY_TENANT_API_KEYS` | *(required)* | JSON map of gateway-issued key → tenant id |
| `GATEWAY_EMBEDDING_MODEL_PATH` | *(unset)* | ONNX embedding model path. Unset disables the semantic cache — never a boot failure |
| `GATEWAY_CACHE_DISTANCE_THRESHOLD` | `0.10` | Cosine distance cutoff for a cache hit |
| `GATEWAY_CACHE_TTL_S` | `3600` | Cache entry TTL |
| `GATEWAY_CACHE_TEMPERATURE_THRESHOLD` | `0.3` | Requests above this (or with `temperature` unset) bypass the cache |
| `GATEWAY_CACHE_LOOKUP_TIMEOUT_MS` | `50` | Bounds the fail-open path if Redis is slow/down |
| `GATEWAY_PROMPT_INJECTION_MODE` | `off` | `off` (detector never runs) / `flag` (detected, logged, counted, request proceeds) / `block` (same as `flag` for now — enforcement is a follow-up, [issue #19](https://github.com/nimrodo/fastapi-ctx-gateway/issues/19)) |
| `GATEWAY_PROMPT_INJECTION_BACKEND` | *(unset)* | `local_classifier` is the only backend today. Unset means no ML backend is registered — **boot fails** if `GATEWAY_PROMPT_INJECTION_MODE` isn't `off`, since a mode the deployer explicitly enabled must not silently become a no-op |
| `GATEWAY_PROMPT_INJECTION_MODEL_PATH` | *(unset)* | ONNX classification model path, required when the backend is `local_classifier`. Same local-model pattern as `GATEWAY_EMBEDDING_MODEL_PATH` — `onnxruntime` is already a core dependency, so no extra install is needed |
| `GATEWAY_PROMPT_INJECTION_THRESHOLD` | `0.5` | Injection-class probability at/above which a request is flagged |
| `GATEWAY_PROMPT_INJECTION_TIMEOUT_MS` | `200` | Bounds the fail-open path if classification is slow/erroring. Larger than the cache's timeout since local model inference is CPU-bound, not network-bound |
| `GATEWAY_TOKEN_BUDGETS` | flash: 32k, pro: 64k | Per-model pruning-trigger caps |
| `GATEWAY_RPM_LIMIT` | `60` | Requests per minute, per tenant+model |
| `GATEWAY_TPM_LIMIT` | `100000` | Tokens per minute, per tenant+model |
| `GATEWAY_RATE_LIMIT_WINDOW_S` | `60` | Rate-limit window size |
| `GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive failures before the breaker opens |
| `GATEWAY_CIRCUIT_BREAKER_RESET_TIMEOUT_S` | `30.0` | Time before a half-open trial request |
| `GATEWAY_HOST` | `0.0.0.0` | Server bind address (CLI only) |
| `GATEWAY_PORT` | `8000` | Server bind port (CLI only) |
| `GATEWAY_WORKERS` | `1` | Worker process count (CLI only) |

## Overriding programmatically

When using the gateway as a library, construct `Settings` directly instead of relying on the environment — handy for tests:

```python
from fastapi_ctx_gateway import Settings, create_app

settings = Settings(
    gemini_upstream_key="test-key",
    tenant_api_keys={"test-key": "test-tenant"},
    cache_ttl_s=60,
)
app = create_app(settings)
```

See the full field list in the [Config reference][fastapi_ctx_gateway.config.Settings].
