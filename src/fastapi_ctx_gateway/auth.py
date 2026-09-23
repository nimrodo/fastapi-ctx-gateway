"""Gateway-issued API key auth.

This authenticates clients to *this* gateway (`x-gateway-api-key`) — a
separate concern from `x-goog-api-key`, which the gateway itself uses to
authenticate to Gemini upstream (see proxy/client.py).
"""

import hmac
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request

from fastapi_ctx_gateway.deps import get_auth_rate_limiter
from fastapi_ctx_gateway.ratelimit import RateLimiter, RateLimitExceeded

__all__ = ["Tenant", "resolve_tenant", "verify_api_key"]


@dataclass(frozen=True)
class Tenant:
    """An authenticated caller of the gateway."""

    id: str
    api_key: str


def resolve_tenant(api_key: str | None, tenant_api_keys: dict[str, str]) -> Tenant:
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
        if hmac.compare_digest(candidate_key.encode(), api_key_bytes):
            matched_tenant_id = tenant_id
    if matched_tenant_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return Tenant(id=matched_tenant_id, api_key=api_key)


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
