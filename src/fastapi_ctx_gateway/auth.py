"""Gateway-issued API key auth.

This authenticates clients to *this* gateway (`x-gateway-api-key`) — a
separate concern from the credentials each provider adapter uses to
authenticate to its own upstream (e.g. `x-goog-api-key` for Gemini; see
`providers/registry.py`).
"""

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request
from pydantic import SecretStr

from fastapi_ctx_gateway.deps import get_auth_rate_limiter
from fastapi_ctx_gateway.ratelimit import RateLimiter, RateLimitExceeded

__all__ = ["Tenant", "resolve_tenant", "verify_api_key"]


@dataclass(frozen=True)
class Tenant:
    """An authenticated caller of the gateway."""

    id: str
    # SecretStr, not str: the dataclass's auto-generated __repr__ would
    # otherwise print the raw gateway API key on any accidental
    # `logger.debug(tenant)`/f"{tenant}" — see #24. Unwrap with
    # `.get_secret_value()` only at the point of use (same convention as
    # the upstream provider keys in providers/registry.py).
    api_key: SecretStr


def resolve_tenant(api_key: str | None, tenant_api_keys: dict[SecretStr, str]) -> Tenant:
    """Look up a Tenant for a gateway API key, or raise 401.

    Compares against every candidate key with `hmac.compare_digest`
    (constant-time per comparison) rather than a plain dict lookup, so an
    attacker can't use response timing to extract a valid key
    character-by-character (see docs/security.md). Always compares
    against every candidate — never short-circuits on the first
    match-shaped key — so total time depends only on how many keys are
    configured, never on how close `api_key` is to any one of them.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    api_key_bytes = api_key.encode()
    matched_tenant_id: str | None = None
    for candidate_key, tenant_id in tenant_api_keys.items():
        if hmac.compare_digest(candidate_key.get_secret_value().encode(), api_key_bytes):
            matched_tenant_id = tenant_id
    if matched_tenant_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return Tenant(id=matched_tenant_id, api_key=SecretStr(api_key))


async def verify_api_key(
    request: Request,
    x_gateway_api_key: str | None = Header(default=None),
    auth_rate_limiter: RateLimiter = Depends(get_auth_rate_limiter),
) -> Tenant:
    """FastAPI dependency resolving the caller's Tenant from the request header.

    A request that resolves to a valid Tenant never touches the failed-auth
    limiter at all — it's already governed by the normal, per-tenant
    `RateLimiter` post-auth, and a valid credential holder must never be
    locked out by noise from other failed attempts sharing their source
    (e.g. a shared/NAT'd IP). Only a failed resolution records against the
    limiter (the gateway's existing `RateLimiter`, reused as a second
    instance keyed per-source-IP under its own settings/namespace — see
    app.py), and once a source's failed-auth budget for its window is
    exhausted, further failed attempts from it get 429 instead of 401 —
    the credential-stuffing/brute-force backstop from docs/security.md.

    Deliberately `resolve_tenant`-then-check, not check-then-`resolve_tenant`:
    `RateLimiter.check()` is an atomic check-*and*-consume with no
    non-consuming peek, so checking first would consume budget on every
    request, including ones that turn out valid — exactly what "only apply
    the check to failure" (docs/security.md) rules out. The cost is that an
    already-exhausted source still pays for one `resolve_tenant` call
    (in-memory, `hmac.compare_digest` over the configured keys) per attempt
    rather than being short-circuited before it — cheap enough not to
    matter, and it keeps a legitimate tenant's own retries never touching
    Redis for this limiter at all.
    """
    settings = request.app.state.settings
    source = request.client.host if request.client else "unknown"
    try:
        return resolve_tenant(x_gateway_api_key, settings.tenant_api_keys)
    except HTTPException:
        decision = await auth_rate_limiter.check(f"authfail:{source}", estimated_tokens=0)
        if not decision.allowed:
            raise RateLimitExceeded(decision) from None
        raise
