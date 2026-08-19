# Installation & quickstart

## Install

```bash
uv add fastapi-ctx-gateway
```

Or, working from a clone of the repo:

```bash
uv sync
```

## Start Redis

The gateway needs a Redis Stack instance (the RediSearch/VSS module) for rate limiting, and optionally for the semantic cache:

```bash
docker compose up -d redis
```

## Configure and run

At minimum, the gateway needs your Gemini API key and at least one gateway-issued key for clients to authenticate with:

```bash
GATEWAY_GEMINI_UPSTREAM_KEY=<your-gemini-api-key> \
GATEWAY_TENANT_API_KEYS='{"my-gateway-key": "my-tenant"}' \
  uv run fastapi-ctx-gateway
```

The server listens on `http://localhost:8000` by default.

!!! tip
    See [Configuration](configuration.md) for every setting, or run `python -m fastapi_ctx_gateway` as an equivalent to the `fastapi-ctx-gateway` script.

## Check it's alive

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}
```

Continue to [Your first request](first-request.md) to actually call Gemini through the gateway.
