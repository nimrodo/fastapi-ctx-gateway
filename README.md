# fastapi-ctx-gateway

A context-aware agentic API gateway for Google's Gemini API. It sits between clients and Gemini, handling stream proxying, semantic caching, token-aware rate limiting, and context pruning at the edge — with a target overhead budget of ~15-20ms on a cache-hit path.

## Architecture

The gateway proxies Gemini's native `streamGenerateContent` wire schema unchanged — no translation layer. Every request is measured against two separate latency budgets:

- **Cache-hit path** (embed + Redis lookup + response synthesis, no Gemini call): target ≤15-20ms total.
- **Cache-miss path** (auth + rate-limit + prune, before the Gemini call): target low single-digit ms, additive to whatever Gemini itself takes.

Request lifecycle: **auth → rate-limit check → circuit-breaker precheck → prune → cache-eligibility → cache lookup → (hit: return) / (miss: proxy, tee, finalize)**. A rejected or short-circuited request never reaches the more expensive steps downstream of it. See `CONTEXT.md` for the domain vocabulary and `docs/adr/` for why each major fork was decided the way it was.

All shared state (rate limits, semantic cache) lives in Redis — the gateway itself is stateless per worker, so it scales horizontally behind a load balancer. The `httpx` client, ONNX embedding session, and Redis connection pool are each built once at startup and shared across every request.

## Quickstart

```bash
uv sync
docker compose up -d redis          # Redis Stack (RediSearch/VSS module)
GATEWAY_GEMINI_UPSTREAM_KEY=<your-gemini-api-key> \
GATEWAY_TENANT_API_KEYS='{"some-gateway-key": "your-tenant-id"}' \
  uv run fastapi-ctx-gateway
```

The server listens on `:8000` by default. Point a client at `POST /v1/{model}:streamGenerateContent` with header `x-gateway-api-key: some-gateway-key` and a normal Gemini request body.

Health check: `GET /healthz`. Prometheus metrics (gateway-specific counters plus generic HTTP metrics): `GET /metrics`.

## Using it as a library

```python
from fastapi_ctx_gateway import create_app, Settings

app = create_app(Settings())  # or mount as a sub-app, or override Settings for tests
```

Individual components — `SemanticCache`, `OnnxVectorizer`, `RateLimiter`, `TokenEstimator`, `TokenBudgetPruner`, `CircuitBreaker` — are also importable standalone (`from fastapi_ctx_gateway import TokenBudgetPruner`), with no FastAPI/routing dependency pulled in. The CLI entrypoint (`fastapi_ctx_gateway.cli.main`) is intentionally not part of this import surface, so importing the package never has the side effect of parsing the environment or being runnable as a server.

A runnable example of mounting the gateway inside a host app lives in [`examples/library_mount/`](examples/library_mount).

## Configuration

All settings are environment variables prefixed `GATEWAY_` (or a `.env` file). Nested fields like `token_budgets` accept JSON.

| Variable | Default | Notes |
|---|---|---|
| `GATEWAY_REDIS_URL` | `redis://localhost:6379` | Shared state store |
| `GATEWAY_GEMINI_UPSTREAM_KEY` | *(required)* | Gemini API key, sent as `x-goog-api-key` |
| `GATEWAY_GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com` | |
| `GATEWAY_TENANT_API_KEYS` | *(required)* | JSON map of gateway-issued key → tenant id |
| `GATEWAY_EMBEDDING_MODEL_PATH` | *(unset)* | Path to an ONNX embedding model. Unset disables the semantic cache entirely — never a boot failure. |
| `GATEWAY_CACHE_DISTANCE_THRESHOLD` | `0.10` | Cosine distance cutoff for a cache hit |
| `GATEWAY_CACHE_TTL_S` | `3600` | Cache entry TTL |
| `GATEWAY_CACHE_TEMPERATURE_THRESHOLD` | `0.3` | Requests above this (or with `temperature` unset) bypass the cache |
| `GATEWAY_CACHE_LOOKUP_TIMEOUT_MS` | `50` | Bounds the fail-open path if Redis is slow/down |
| `GATEWAY_TOKEN_BUDGETS` | `{"gemini-3.7-flash": 32000, "gemini-2.5-pro": 64000}` | Per-model pruning-trigger caps; `default` key sets the fallback |
| `GATEWAY_RPM_LIMIT` / `GATEWAY_TPM_LIMIT` | `60` / `100000` | Per tenant+model, per `GATEWAY_RATE_LIMIT_WINDOW_S` |
| `GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive failures before the breaker opens |
| `GATEWAY_CIRCUIT_BREAKER_RESET_TIMEOUT_S` | `30.0` | Time before a half-open trial request |
| `GATEWAY_HOST` / `GATEWAY_PORT` / `GATEWAY_WORKERS` | `0.0.0.0` / `8000` / `1` | Server bind settings (CLI only) |

The semantic cache uses a placeholder tokenizer (`simple_char_code_tokenize`) until a real embedding model is deployed — see the "Embedding model" section below.

## Local development

```bash
uv sync
docker compose up -d redis          # needed for integration tests and real cache/rate-limit behavior
uv run ruff check .                 # includes docstring lint on the public API
uv run ruff format --check .
uv run ty check
uv run pytest -x -q                 # unit tests: fast, no external services
uv run pytest -m integration -q     # requires the Redis Stack container above
uv run pytest -m slow -q            # packaging smoke test + latency benchmark, opt-in
uv run pytest --cov=fastapi_ctx_gateway --cov-report=term-missing
```

## Embedding model

The semantic cache needs an ONNX embedding model to be useful in production. `GATEWAY_EMBEDDING_MODEL_PATH` unset (the default) disables the cache — every request is a guaranteed miss, and the gateway still boots and serves traffic normally. Point it at a real exported model (e.g. a quantized `all-MiniLM-L6-v2`) to enable it. Test fixtures use a tiny synthetic ONNX model (`tests/fixtures/generate_tiny_onnx_model.py`) — not a real embedding model, just enough structure to exercise the pipeline offline.

## Repository layout

- `src/fastapi_ctx_gateway/` — the package (see module docstrings for what each file owns)
- `tests/unit/` — fast, fully mocked
- `tests/integration/` — real Redis Stack + mocked Gemini (`respx`)
- `CONTEXT.md` — domain glossary
- `docs/adr/` — architecture decision records for the major forks in this design
- `docs/agents/` — configuration consumed by AI coding-agent skills working in this repo (issue tracker, triage labels, domain-doc conventions)
