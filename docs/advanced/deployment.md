# Deployment

## Local development

`docker-compose.yml` at the repo root runs a Redis Stack container (the RediSearch/VSS module required for rate limiting and the semantic cache):

```bash
docker compose up -d redis
```

## Running the server

```bash
uv run fastapi-ctx-gateway
# equivalently:
uv run python -m fastapi_ctx_gateway
```

Reads all configuration from the environment (see [Configuration](../tutorial/configuration.md)). `GATEWAY_HOST`, `GATEWAY_PORT`, and `GATEWAY_WORKERS` control the bind address and process count.

## Multiple workers

`GATEWAY_WORKERS` maps directly to `uvicorn.run(..., workers=N)`. Every worker builds its own singletons independently at startup — the `httpx` client, Redis connection pool, ONNX embedding session, and (crucially) the [circuit breaker](../tutorial/circuit-breaker.md), which is deliberately per-worker and in-memory rather than synchronized across workers. Rate limiting and the semantic cache remain correct across workers because their state lives in Redis, not in-process.

## Production checklist

- [ ] `GATEWAY_GEMINI_UPSTREAM_KEY` set from a secrets manager, not a plain env var in source control.
- [ ] `GATEWAY_REDIS_URL` points at a Redis Stack instance with persistence/HA appropriate for your rate-limit and cache data (losing it just means budgets/cache reset, not data-loss in the traditional sense — but a shared cluster instance beats a single ephemeral container).
- [ ] `GATEWAY_EMBEDDING_MODEL_PATH` set to a real exported model if you want the semantic cache enabled — see [Custom vectorizer](custom-vectorizer.md).
- [ ] `GATEWAY_TENANT_API_KEYS` populated with real, unique per-tenant keys — see [Multi-tenant keys](multi-tenant-keys.md).
- [ ] `OTEL_*` environment variables configured if you want spans exported somewhere real — see [Observability](../tutorial/observability.md).
- [ ] `/metrics` scraped by your Prometheus (or compatible) setup.
- [ ] Rate-limit and circuit-breaker thresholds tuned for your actual Gemini quota and traffic patterns, not left at the conservative defaults.

## Health checks

`GET /healthz` returns `{"status": "ok"}` unconditionally once the app has started — it doesn't probe Redis or Gemini (the gateway is designed to boot and serve even with the cache disabled or Redis briefly unreachable, so a health check that failed on those would misrepresent the gateway's actual availability).
