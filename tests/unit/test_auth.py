"""Tests for gateway API-key auth."""

import pytest
from fastapi import HTTPException

from fastapi_ctx_gateway.auth import Tenant, resolve_tenant


def test_resolve_tenant_valid_key() -> None:
    tenant = resolve_tenant("secret-1", tenant_api_keys={"secret-1": "tenant-a"})
    assert tenant == Tenant(id="tenant-a", api_key="secret-1")


def test_resolve_tenant_invalid_key_raises_401() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_tenant("wrong-key", tenant_api_keys={"secret-1": "tenant-a"})
    assert exc_info.value.status_code == 401
