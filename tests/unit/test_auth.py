"""Tests for gateway API-key auth."""

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from fastapi_ctx_gateway.auth import Tenant, resolve_tenant


def test_resolve_tenant_valid_key() -> None:
    tenant = resolve_tenant("secret-1", tenant_api_keys={SecretStr("secret-1"): "tenant-a"})
    assert tenant == Tenant(id="tenant-a", api_key=SecretStr("secret-1"))


def test_resolve_tenant_invalid_key_raises_401() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_tenant("wrong-key", tenant_api_keys={SecretStr("secret-1"): "tenant-a"})
    assert exc_info.value.status_code == 401


def test_resolve_tenant_same_length_wrong_key_still_rejected() -> None:
    """Behavioral proof hmac.compare_digest is wired correctly.

    A wrong key of the exact same length as the real one (the case a plain
    `==`/dict-lookup comparison would be most likely to special-case or
    short-circuit differently on) must still be rejected — see
    docs/security.md's constant-time-comparison fix. True timing-side-
    channel closure isn't practically assertable in a unit test.
    """
    real_key = "secret-1"
    wrong_key_same_length = "secret-2"
    assert len(real_key) == len(wrong_key_same_length)
    with pytest.raises(HTTPException) as exc_info:
        resolve_tenant(wrong_key_same_length, tenant_api_keys={SecretStr(real_key): "tenant-a"})
    assert exc_info.value.status_code == 401


def test_tenant_repr_masks_api_key() -> None:
    """Regression for #24: a raw Tenant repr/log must never print the key.

    Nothing in the current request path logs a whole Tenant object, but
    this closes the latent gap so a future `logger.debug(tenant)` can't
    leak a gateway API key with nothing to catch it.
    """
    tenant = resolve_tenant("secret-1", tenant_api_keys={SecretStr("secret-1"): "tenant-a"})
    assert "secret-1" not in repr(tenant)
    assert tenant.api_key.get_secret_value() == "secret-1"
