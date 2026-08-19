"""Tests for the CLI entrypoint."""

from unittest.mock import MagicMock

from fastapi_ctx_gateway import cli


def test_main_builds_app_and_runs_uvicorn(monkeypatch) -> None:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"test-key":"test-tenant"}')
    run_mock = MagicMock()
    monkeypatch.setattr(cli.uvicorn, "run", run_mock)

    cli.main()

    assert run_mock.call_count == 1
    _, kwargs = run_mock.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000
