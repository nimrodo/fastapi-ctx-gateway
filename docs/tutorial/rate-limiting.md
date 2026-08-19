# Rate limiting

Every request is checked against a token (TPM) and request (RPM) budget, keyed by `{tenant_api_key}:{model}` — so a noisy tenant on one model never starves their own budget on another, and tenants never share a budget with each other.

## How the check works

1. A cheap local heuristic estimates the request's token count *before* calling Gemini (`len(text) // 4` per text part, plus a small per-turn overhead, biased slightly high). Real tokenization would mean a second network round trip to Gemini just to count tokens — too slow for the admission-control hot path.
2. That estimate is checked against the budget with a single atomic Redis round trip (a Lua script, `EVALSHA`) using a sliding-window-*counter* approximation, not an exact log — see [ADR-0004](../adr/0004-sliding-window-counter-over-exact-log.md) for why.
3. If rejected, the client gets `429` with a `Retry-After` header computed from how far into the current window it is.
4. If admitted and the request succeeds, the counter is **reconciled** afterward against Gemini's real `usageMetadata` — the estimate was only ever provisional.

```bash
curl -i -X POST http://localhost:8000/v1/gemini-3.7-flash:streamGenerateContent \
  -H "x-gateway-api-key: my-gateway-key" -d '{"contents": []}'
# HTTP/1.1 429 Too Many Requests
# Retry-After: 12
```

## Tuning

```bash
GATEWAY_RPM_LIMIT=120
GATEWAY_TPM_LIMIT=200000
GATEWAY_RATE_LIMIT_WINDOW_S=60
```

These are global defaults today (see [`Settings`][fastapi_ctx_gateway.config.Settings]); per-tenant overrides are a natural next step if different tenants need different budgets — the `{tenant}:{model}` key already supports it, only the config source would need to change.

## Ordering

Rate limiting runs immediately after auth, before the circuit-breaker check, pruning, or cache lookup — a rejected request should do as little work as possible before being turned away.
