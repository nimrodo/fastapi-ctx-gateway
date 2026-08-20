# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                       # install/sync dependencies

uv run ruff check .                           # lint (includes docstring lint on the public API)
uv run ruff format --check .                  # format check (drop --check to auto-format)
uv run ty check                                # type check

docker compose up -d redis                    # Redis Stack (RediSearch/VSS) — needed for integration tests
uv run pytest -x -q                           # unit tests: fast, no external services
uv run pytest -m integration -q               # integration tests: requires the Redis Stack container above
uv run pytest -m slow -q                      # packaging smoke test + latency benchmark, opt-in
uv run pytest path/to/test_file.py::test_name # a single test
uv run pytest --cov=fastapi_ctx_gateway --cov-report=term-missing

uv run mkdocs build --strict                  # docs site; fails on any broken link/mkdocstrings reference

GATEWAY_GEMINI_UPSTREAM_KEY=<key> GATEWAY_TENANT_API_KEYS='{"k":"t"}' uv run fastapi-ctx-gateway  # run the server
```

CI (`.github/workflows/ci.yml`) runs three jobs on every push/PR: `lint-and-unit` (ruff + ty + `pytest -m "not integration and not slow"`, no Redis), `integration` (`pytest -m integration` against a real `redis/redis-stack-server` service container), `docs-build` (`mkdocs build --strict`). Any new file under `tests/integration/` **must** set `pytestmark = pytest.mark.integration` — the split relies on that marker, not the directory name, and a missing marker fails the no-Redis job.

## Architecture

**Neutral contract + pluggable providers.** The gateway does not speak any single upstream's wire format. `schemas/neutral.py` defines the gateway's own request/response/error contract (`NeutralGenerateRequest`, `Turn`/`Part`, `NeutralStreamEvent`, `NeutralErrorEvent`); a `Provider` (`providers/base.py`) translates that contract to/from one upstream API. `GeminiProvider` and `OpenAIProvider` are the two implementations — read both before adding a third, and see `docs/advanced/adding-a-provider.md` for the full walkthrough. Route shape: `POST /v1/{provider}/{model}:streamGenerateContent`, where `{provider}` selects the `Provider` from `app.state.providers` (a plain `dict[str, Provider]`).

**Request lifecycle** (`routers/generate.py`, the only router): auth → rate-limit check → circuit-breaker precheck → prune → cache-eligibility → cache lookup → (hit: synthesize a response from the cache) / (miss: call `provider.stream()`, tee the bytes to the client while accumulating them, then finalize). A rejected or short-circuited request never reaches the more expensive steps downstream of it — admission control runs against the *unpruned* token estimate, but the cache is keyed against *pruned* turns.

**Everything is per-provider, not global.** Each registered provider gets its own `CircuitBreaker` (`app.state.circuit_breakers`, keyed by provider name) and its own `circuit_breaker_open_total{provider}` metric — an outage on one upstream must never short-circuit an unrelated one. `RateLimiter` is keyed by `{tenant_api_key}:{model}`, not by provider. `SemanticCache` is keyed by `{tenant_id, model}` tags.

**Provider registration is config-driven.** Gemini is mandatory (`GATEWAY_GEMINI_UPSTREAM_KEY` has no default; `Settings()` refuses to boot without it). Every other provider is an *optional config group* instead: unset (or empty-string) credentials mean that provider simply isn't registered — `/v1/{name}/...` 404s — never a boot failure. `app.py::_registered_provider_names()` is the single source of truth for which providers exist; it drives both the provider registry and the circuit-breaker registry.

**`Provider.stream()` never raises.** Upstream/transport failures are translated in-band to a terminal `NeutralErrorEvent` SSE chunk (via the shared `providers/sse.py` helpers: `iter_sse_data_lines`, `neutral_error_event`, `parse_error_message`). Each provider implements its own bounded pre-stream retry (one retry, only before any bytes reach the client) — this used to live in the generic router-level `stream_with_retry`, but moved into the provider once `stream()` stopped raising.

**Fail-open, not fail-closed, for optional subsystems.** The semantic cache (`cache/semantic_cache.py`) and its embedding model are optional — any lookup/store failure, timeout, or missing model degrades to "cache disabled" / "this request is a miss," never a request failure. This pattern (never let an optimization become a hard dependency) is the same reasoning behind the per-provider circuit breaker and the optional-provider-config pattern above.

**No module-level app instance.** Every caller (`cli.py`, tests, a consumer mounting the gateway as a library sub-app) builds its own app via `create_app(settings)` — see `docs/tutorial/using-as-a-library.md` and `examples/library_mount/`.

For the domain vocabulary (Tenant, Hit path/Miss path, Cache eligibility, Finished cleanly, etc.) see `CONTEXT.md`. For why each major architectural fork was decided the way it was, see `docs/adr/` (six ADRs as of this writing; ADR-0006 covers the neutral-schema/provider-abstraction decision specifically).

## Git workflow

Work on feature branches and open PRs into `main` — do not commit directly to `main`.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (nimrodo/fastapi-ctx-gateway), using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
