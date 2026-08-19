# Multi-tenant keys

Clients authenticate to the gateway with a **gateway-issued** key, not your Gemini key — the gateway holds the real Gemini key once and never exposes it downstream.

## Static configuration

Today, the mapping from gateway key to tenant id is static config:

```bash
GATEWAY_TENANT_API_KEYS='{"key-for-team-a": "team-a", "key-for-team-b": "team-b"}'
```

Resolved by [`resolve_tenant`][fastapi_ctx_gateway.auth.resolve_tenant] into a [`Tenant`][fastapi_ctx_gateway.auth.Tenant] — a missing or unrecognized key gets `401` before any other work happens.

## Why tenant id, not the raw key, flows through the system

Rate-limit keys, cache partitions, and reconciliation are all keyed off `tenant.id` combined with the model name (`{tenant_api_key}:{model}` for rate limiting; `tenant_id`/`model` tags for the cache) — never the raw key value directly beyond that first resolution step. This means rotating a tenant's gateway key doesn't require migrating their rate-limit or cache state, as long as the tenant id in `GATEWAY_TENANT_API_KEYS` stays the same.

## Moving beyond static config

The static map is a deliberate v1 simplification — swap [`auth.py`][fastapi_ctx_gateway.auth] for a Redis-backed or database-backed lookup once onboarding needs to be dynamic (self-service key generation, revocation without a redeploy, per-tenant rate-limit overrides). The `Tenant` dataclass and `resolve_tenant()` signature are the seam to extend; nothing downstream of auth needs to change.
