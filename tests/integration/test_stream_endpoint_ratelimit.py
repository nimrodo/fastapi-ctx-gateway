"""Router-level test: rate-limit rejection short-circuits before proxying to Gemini."""

import httpx
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.deps import get_rate_limiter
from fastapi_ctx_gateway.ratelimit import RateLimitDecision


class _AlwaysRejectLimiter:
    async def check(self, key: str, estimated_tokens: int) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=False, retry_after_s=12.0, weighted_tokens=0, weighted_requests=0
        )

    async def reconcile(self, key: str, estimated_tokens: int, actual_tokens: int) -> None:
        raise AssertionError("reconcile should never be called for a rejected request")


class _AlwaysAllowLimiter:
    def __init__(self) -> None:
        self.reconcile_calls: list[tuple[str, int, int]] = []

    async def check(self, key: str, estimated_tokens: int) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=True, retry_after_s=0.0, weighted_tokens=0, weighted_requests=0
        )

    async def reconcile(self, key: str, estimated_tokens: int, actual_tokens: int) -> None:
        self.reconcile_calls.append((key, estimated_tokens, actual_tokens))


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    return Settings()


def test_rate_limit_rejection_returns_429_and_never_calls_gemini(monkeypatch) -> None:
    with respx.mock(
        base_url="https://generativelanguage.googleapis.com", assert_all_called=False
    ) as mock:
        route = mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n")
        )

        app = create_app(_settings(monkeypatch))
        app.dependency_overrides[get_rate_limiter] = lambda: _AlwaysRejectLimiter()
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "12"
    assert route.call_count == 0


def test_successful_stream_reconciles_actual_token_usage(monkeypatch) -> None:
    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},'
        b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":42}}\n\n'
    )
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        limiter = _AlwaysAllowLimiter()
        app.dependency_overrides[get_rate_limiter] = lambda: limiter
        with TestClient(app) as client:
            response = client.post(
                "/v1/gemini-2.5-flash:streamGenerateContent",
                headers={"x-gateway-api-key": "gw-secret"},
                json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            )

    assert response.status_code == 200
    assert len(limiter.reconcile_calls) == 1
    key, estimated, actual = limiter.reconcile_calls[0]
    assert key == "gw-secret:gemini-2.5-flash"
    assert actual == 42
