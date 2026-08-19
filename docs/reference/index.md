# Reference

Auto-generated from the docstrings in `src/fastapi_ctx_gateway/` — every public class and function is documented at its source, enforced by `ruff`'s docstring lint since the project's first commit, so this reference never drifts silently out of sync with the code.

Start with [App](app.md) for `create_app()` and [Config](config.md) for `Settings`, or jump straight to the component you need:

- [Cache](cache.md) — `SemanticCache`, `OnnxVectorizer`, eligibility
- [Rate limiting](ratelimit.md) — `RateLimiter`, `TokenEstimator`
- [Pruning](pruning.md) — `TokenBudgetPruner`
- [Circuit breaker](circuit-breaker.md) — `CircuitBreaker`, `CircuitState`
- [Proxy](proxy.md) — `GeminiClient`, streaming/retry helpers
- [Schemas](schemas.md) — the Gemini wire-schema models
- [Auth](auth.md) — `Tenant`, `resolve_tenant`
- [Observability](observability.md) — metrics and tracing helpers
