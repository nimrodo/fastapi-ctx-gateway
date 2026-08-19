"""Gateway-issued API key auth.

This authenticates clients to *this* gateway (`x-gateway-api-key`) — a
separate concern from `x-goog-api-key`, which the gateway itself uses to
authenticate to Gemini upstream (see proxy/client.py).
"""

from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

__all__ = ["Tenant", "resolve_tenant", "verify_api_key"]


@dataclass(frozen=True)
class Tenant:
    """An authenticated caller of the gateway."""

    id: str
    api_key: str


def resolve_tenant(api_key: str | None, tenant_api_keys: dict[str, str]) -> Tenant:
    """Look up a Tenant for a gateway API key, or raise 401."""
    if api_key is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    tenant_id = tenant_api_keys.get(api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return Tenant(id=tenant_id, api_key=api_key)


async def verify_api_key(
    request: Request, x_gateway_api_key: str | None = Header(default=None)
) -> Tenant:
    """FastAPI dependency resolving the caller's Tenant from the request header."""
    tenant_api_keys = request.app.state.settings.tenant_api_keys
    return resolve_tenant(x_gateway_api_key, tenant_api_keys)
