"""Integration test: the failed-auth backstop (real Redis-backed RateLimiter)."""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

pytestmark = pytest.mark.integration


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    # A tight threshold so the test doesn't need dozens of requests.
    monkeypatch.setenv("GATEWAY_AUTH_FAILURE_RPM_LIMIT", "3")
    monkeypatch.setenv("GATEWAY_AUTH_FAILURE_WINDOW_S", "60")
    return Settings()


def _post(client: TestClient, api_key: str) -> httpx.Response:
    return client.post(
        "/v1/gemini/gemini-3.7-flash:streamGenerateContent",
        headers={"x-gateway-api-key": api_key},
        json={"turns": [{"role": "user", "parts": [{"type": "text", "text": "hi"}]}]},
    )


def test_repeated_failed_auth_attempts_past_threshold_are_rejected(
    monkeypatch, redis_client
) -> None:
    """Once a source exhausts its failed-auth budget, further *wrong-key*
    attempts from it get 429 instead of 401 — the credential-stuffing /
    brute-force backstop.
    """
    with respx.mock(
        base_url="https://generativelanguage.googleapis.com", assert_all_called=False
    ) as mock:
        route = mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n")
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            for _ in range(3):
                response = _post(client, "wrong-key")
                assert response.status_code == 401
            still_wrong = _post(client, "another-wrong-key")

    assert still_wrong.status_code == 429
    assert route.call_count == 0  # never reached Gemini


def test_valid_key_sharing_a_source_with_prior_failures_is_not_itself_blocked(
    monkeypatch, redis_client
) -> None:
    """A tenant who mistypes their key a few times — even past the failed-
    auth threshold — shouldn't lock themselves out of their next, correct
    attempt: a valid key never touches the failed-auth limiter at all.
    """
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-3.7-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=b"data: {}\n\n", headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            for _ in range(3):
                response = _post(client, "wrong-key")
                assert response.status_code == 401
            ok = _post(client, "gw-secret")

    assert ok.status_code == 200
