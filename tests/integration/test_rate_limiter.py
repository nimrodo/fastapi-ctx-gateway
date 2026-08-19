"""Integration tests for the Lua-backed sliding-window rate limiter (real Redis)."""

import pytest

from fastapi_ctx_gateway.ratelimit import RateLimiter

pytestmark = pytest.mark.integration


@pytest.fixture
def limiter(redis_client) -> RateLimiter:
    return RateLimiter(redis_client=redis_client, rpm_limit=3, tpm_limit=1_000, window_s=60)


async def test_requests_within_budget_are_allowed(limiter: RateLimiter) -> None:
    for _ in range(3):
        decision = await limiter.check("tenant-a:gemini-2.5-flash", estimated_tokens=10)
        assert decision.allowed is True


async def test_request_over_rpm_budget_is_rejected(limiter: RateLimiter) -> None:
    for _ in range(3):
        await limiter.check("tenant-a:gemini-2.5-flash", estimated_tokens=10)
    decision = await limiter.check("tenant-a:gemini-2.5-flash", estimated_tokens=10)
    assert decision.allowed is False
    assert decision.retry_after_s >= 1.0


async def test_request_over_tpm_budget_is_rejected(redis_client) -> None:
    limiter = RateLimiter(redis_client=redis_client, rpm_limit=100, tpm_limit=50, window_s=60)
    decision = await limiter.check("tenant-b:gemini-2.5-flash", estimated_tokens=60)
    assert decision.allowed is False


async def test_rejected_requests_do_not_consume_budget(limiter: RateLimiter) -> None:
    for _ in range(3):
        await limiter.check("tenant-c:gemini-2.5-flash", estimated_tokens=10)
    # Rejected — should not further increase the counters.
    await limiter.check("tenant-c:gemini-2.5-flash", estimated_tokens=10)
    decision = await limiter.check("tenant-c:gemini-2.5-flash", estimated_tokens=10)
    assert decision.allowed is False
    assert decision.weighted_requests < 5  # still ~3, not 4-5


async def test_different_keys_have_independent_budgets(limiter: RateLimiter) -> None:
    for _ in range(3):
        await limiter.check("tenant-d:gemini-2.5-flash", estimated_tokens=10)
    decision = await limiter.check("tenant-e:gemini-2.5-flash", estimated_tokens=10)
    assert decision.allowed is True


async def test_reconcile_adjusts_token_counter(limiter: RateLimiter) -> None:
    key = "tenant-f:gemini-2.5-flash"
    await limiter.check(key, estimated_tokens=100)
    await limiter.reconcile(key, estimated_tokens=100, actual_tokens=40)
    # Budget was consumed at the (lower) reconciled amount, so more requests fit.
    decision = await limiter.check(key, estimated_tokens=800)
    assert decision.allowed is True
