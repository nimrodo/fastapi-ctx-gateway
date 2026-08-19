# Example: mounting the gateway as a library

Demonstrates the pattern from [Using it as a library](../../docs/tutorial/using-as-a-library.md):
a host FastAPI app with its own routes mounts `create_app(Settings())` as a sub-app.

## Run it

```bash
# Optional: Redis is only needed once a request actually reaches the gateway
# (rate limiting, pruning, caching). The app boots fine without it.
docker compose up -d

export GATEWAY_GEMINI_UPSTREAM_KEY=your-real-gemini-key

# Maps gateway-issued keys (what clients send in x-gateway-api-key) to a
# tenant id. Required — with no entries, every request gets 401.
export GATEWAY_TENANT_API_KEYS='{"my-gateway-key":"local-dev"}'

uv run uvicorn examples.library_mount.app:app --app-dir . --reload
```

Then, from another terminal:

```bash
curl http://localhost:8000/                # the host app's own route
curl http://localhost:8000/gateway/healthz # the mounted gateway
```

`POST /gateway/v1/gemini-2.5-flash:streamGenerateContent` (with an
`x-gateway-api-key: my-gateway-key` header) behaves exactly like the
standalone gateway (see the [tutorial](../../docs/tutorial/first-request.md)) —
it just lives under a `/gateway` prefix here instead of at the root.

## OpenAPI docs

A mounted sub-app keeps its own independent OpenAPI schema — it does not appear
inside the host app's `/docs`. The host app's Swagger UI at
`http://localhost:8000/docs` only lists the host's own routes (`/`); the
gateway's routes have their own docs at:

```
http://localhost:8000/gateway/docs
http://localhost:8000/gateway/openapi.json
```
