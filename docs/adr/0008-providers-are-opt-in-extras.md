# ADR-0008: Every provider is an opt-in extra; no provider is mandatory

## Context

[ADR-0006](0006-neutral-schema-and-provider-abstraction.md) made adding a provider additive, but left Gemini privileged: `Settings()` refused to boot without `GATEWAY_GEMINI_UPSTREAM_KEY`, `app.py` imported `GeminiProvider` and `OpenAIProvider` at module load, and `_registered_provider_names()` hard-coded Gemini as always-on.

Adding the Anthropic provider ([ADR-0007](0007-anthropic-sdk-provider.md)) brought a real third-party dependency (`anthropic`, which itself pulls `httpx2` and more). Forcing that onto every install — including deployments that only call Gemini or OpenAI — is wrong, and the asymmetry (one mandatory provider, the rest optional-by-config-only) had no principled basis once there were three.

## Decision

Every provider is an **opt-in dependency extra** and an **optional config group**. None is mandatory.

- `pyproject.toml` declares `[gemini]`, `[openai]`, `[anthropic]`, and `[all]` extras. `[gemini]` and `[openai]` are empty today (they need nothing beyond core `httpx`) but exist for install-time symmetry and so a future SDK swap has a home; `[anthropic]` supplies `anthropic`.
- `gemini_upstream_key` is now `SecretStr | None = None`, matching `openai_api_key` / `anthropic_api_key`. A blank string is treated as unset for all three.
- `fastapi_ctx_gateway.providers.registry` is the single source of truth for which providers exist. It holds one `ProviderSpec` per provider — `name`, `extra`, dotted `module` path, a `configured(settings)` predicate, and a `build(settings, http_client)` factory. It imports **no** provider adapter module at load time; `app.py` imports none either.
- `ProviderSpec.build()` does the adapter import lazily and maps `ImportError` to an actionable `RuntimeError` ("install fastapi-ctx-gateway[<extra>]") — a configured key with the extra missing fails loudly, it does not silently 404.
- `create_app()` raises if **zero** providers are configured — a gateway that can serve no traffic should fail at boot, not per request.

## Consequences

- **`import fastapi_ctx_gateway.app` works with no provider extra installed.** Only the neutral contract, pipeline, and registry are imported eagerly.
- **`_registered_provider_names()` / `_build_providers()` are registry-driven** — a one-liner over `SPECS`. Adding a provider is now: write the adapter, add a `ProviderSpec` to `registry.py`. The circuit-breaker registry, `/v1/{name}/...` 404-when-unconfigured, and the `circuit_breaker_open_total{provider}` label all still follow from `_registered_provider_names()`.
- **Bare `pip install fastapi-ctx-gateway` has no providers.** Docs lead with `fastapi-ctx-gateway[all]` (or a single `[gemini]` / `[openai]` / `[anthropic]`); the zero-provider boot check is the guardrail.
- **This amends ADR-0006's "Gemini's mandatory key" consequence.** Gemini is now exactly as optional as the others. Existing deployments that set `GATEWAY_GEMINI_UPSTREAM_KEY` are unaffected.
- **CI and contributors** get `anthropic` via the `dev` dependency group (`uv sync` installs it by default), so lint, type-check, unit tests, and `mkdocs` all exercise the Anthropic adapter without a separate `--extra` step.
