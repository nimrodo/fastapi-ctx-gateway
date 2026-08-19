# Configuration

The gateway is configured entirely through environment variables, prefixed `GATEWAY_`, loaded via [`Settings`][fastapi_ctx_gateway.config.Settings] (a `pydantic-settings` model). A `.env` file in the working directory is picked up automatically.

## Required

```bash
GATEWAY_GEMINI_UPSTREAM_KEY=<your-gemini-api-key>
GATEWAY_TENANT_API_KEYS='{"<gateway-issued-key>": "<tenant-id>"}'
```

`GATEWAY_TENANT_API_KEYS` has no default and the app refuses to boot without it — an empty mapping would mean no client can ever authenticate, so an unconfigured gateway fails fast at startup instead of shipping a service that 401s every request. See [Multi-tenant keys](../advanced/multi-tenant-keys.md). Everything else has a working default.

## Nested settings

Fields like `token_budgets` accept JSON directly:

```bash
GATEWAY_TOKEN_BUDGETS='{"budgets": {"gemini-3.7-flash": 32000, "gemini-2.5-pro": 64000}, "default": 16000}'
```

## Full reference

| Variable | Default | What it controls |
|---|---|---|
| `GATEWAY_REDIS_URL` | `redis://localhost:6379` | Shared state store (rate limits, semantic cache) |
| `GATEWAY_GEMINI_UPSTREAM_KEY` | *(required)* | Sent to Gemini as `x-goog-api-key` |
| `GATEWAY_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | |
| `GATEWAY_TENANT_API_KEYS` | *(required)* | JSON map of gateway-issued key → tenant id |
| `GATEWAY_EMBEDDING_MODEL_PATH` | *(unset)* | ONNX embedding model path. Unset disables the semantic cache — never a boot failure |
| `GATEWAY_CACHE_DISTANCE_THRESHOLD` | `0.10` | Cosine distance cutoff for a cache hit |
| `GATEWAY_CACHE_TTL_S` | `3600` | Cache entry TTL |
| `GATEWAY_CACHE_TEMPERATURE_THRESHOLD` | `0.3` | Requests above this (or with `temperature` unset) bypass the cache |
| `GATEWAY_CACHE_LOOKUP_TIMEOUT_MS` | `50` | Bounds the fail-open path if Redis is slow/down |
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
