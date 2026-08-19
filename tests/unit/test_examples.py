"""Tests for the mount-as-library example under examples/library_mount/."""

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

_APP_PATH = (Path(__file__).parents[2] / "examples" / "library_mount" / "app.py").resolve()


def _load_example_app(monkeypatch):
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    spec = importlib.util.spec_from_file_location("library_mount_app", _APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def test_host_route_is_independent_of_the_mounted_gateway(monkeypatch) -> None:
    app = _load_example_app(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "host app; gateway mounted at /gateway"}


def test_gateway_submount_is_reachable(monkeypatch) -> None:
    app = _load_example_app(monkeypatch)
    with TestClient(app) as client:
        response = client.get("/gateway/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_mounted_gateways_lifespan_actually_runs(monkeypatch) -> None:
    """A mounted sub-app's lifespan doesn't run automatically in Starlette —
    the host app must explicitly propagate it. Without that, app.state on the
    gateway (providers, rate_limiter, etc.) is never populated and every
    real request 500s with a KeyError, even though /healthz (which touches no
    state) looks fine.
    """
    app = _load_example_app(monkeypatch)
    mount = next(route for route in app.routes if getattr(route, "path", None) == "/gateway")
    gateway_app = mount.app
    with TestClient(app):
        assert getattr(gateway_app.state, "providers", None)
        assert getattr(gateway_app.state, "rate_limiter", None) is not None
