# Example: mounting the gateway as a library

Demonstrates the pattern from [Using it as a library](../../docs/tutorial/using-as-a-library.md):
a host FastAPI app with its own routes mounts `create_app(Settings())` as a sub-app.

## Run it

```bash
# Optional: Redis is only needed once a request actually reaches the gateway
# (rate limiting, pruning, caching). The app boots fine without it.
docker compose up -d

export GATEWAY_GEMINI_UPSTREAM_KEY=your-real-or-placeholder-key
uv run uvicorn examples.library_mount.app:app --app-dir . --reload
```

Then, from another terminal:

```bash
curl http://localhost:8000/                # the host app's own route
curl http://localhost:8000/gateway/healthz # the mounted gateway
```

`GET /gateway/gemini-2.5-flash:streamGenerateContent` behaves exactly like the
standalone gateway (see the [tutorial](../../docs/tutorial/first-request.md)) —
it just lives under a `/gateway` prefix here instead of at the root.
