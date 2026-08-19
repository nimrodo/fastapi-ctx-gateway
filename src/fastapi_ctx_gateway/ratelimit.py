"""Token-aware rate limiting: pre-call estimation + a Redis-backed sliding-window limiter."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi_ctx_gateway.schemas.gemini import Content

__all__ = ["RateLimitDecision", "RateLimitExceeded", "RateLimiter", "TokenEstimator"]

_SCRIPT_PATH = Path(__file__).parent / "scripts" / "sliding_window.lua"


class TokenEstimator:
    """A cheap, dependency-light pre-call token estimate.

    Deliberately not a real tokenizer: this only needs to be a fast,
    conservative proxy for admission control. Actual usage is reconciled
    from Gemini's real usageMetadata once the response completes.
    """

    _CHARS_PER_TOKEN = 4
    _PER_TURN_OVERHEAD_TOKENS = 4
    _SAFETY_FACTOR = 1.1

    def estimate(self, contents: list[Content], system_instruction: Content | None) -> int:
        """Estimate prompt tokens over text parts only, biased slightly high."""
        turns = list(contents)
        if system_instruction is not None:
            turns = [*turns, system_instruction]

        total_chars = sum(
            len(part.text) for content in turns for part in content.parts if part.text
        )
        raw = total_chars / self._CHARS_PER_TOKEN + len(turns) * self._PER_TURN_OVERHEAD_TOKENS
        return math.ceil(raw * self._SAFETY_FACTOR)


@dataclass(frozen=True)
class RateLimitDecision:
    """The outcome of a rate-limit admission check."""

    allowed: bool
    retry_after_s: float
    weighted_tokens: float
    weighted_requests: float


class RateLimitExceeded(Exception):
    """Raised when a request is rejected by the rate limiter."""

    def __init__(self, decision: RateLimitDecision) -> None:
        """Carry the rejecting decision so a handler can derive Retry-After."""
        self.decision = decision
        super().__init__(f"rate limit exceeded, retry after {decision.retry_after_s}s")


class RateLimiter:
    """A `{tenant}:{model}`-keyed sliding-window-counter limiter, one Redis round trip per call."""

    def __init__(
        self, redis_client: Any, rpm_limit: int, tpm_limit: int, window_s: int = 60
    ) -> None:
        """Configure limits; the Lua script is loaded lazily on first use."""
        self._redis = redis_client
        self._rpm_limit = rpm_limit
        self._tpm_limit = tpm_limit
        self._window_s = window_s
        self._script_sha: str | None = None

    async def _get_script_sha(self) -> str:
        sha = self._script_sha
        if sha is None:
            sha = await self._redis.script_load(_SCRIPT_PATH.read_text())
            self._script_sha = sha
        return sha

    async def check(self, key: str, estimated_tokens: int) -> RateLimitDecision:
        """Atomically check-and-consume budget for one request. One round trip."""
        sha = await self._get_script_sha()
        now_ms = int(time.time() * 1000)
        allowed, weighted_tokens, weighted_requests, elapsed_ms = await self._redis.evalsha(
            sha,
            1,
            key,
            now_ms,
            self._window_s,
            self._rpm_limit,
            self._tpm_limit,
            estimated_tokens,
            "check",
            0,
        )
        retry_after_s = max(1.0, round(self._window_s - elapsed_ms / 1000, 1))
        return RateLimitDecision(
            allowed=bool(allowed),
            retry_after_s=retry_after_s if not allowed else 0.0,
            weighted_tokens=float(weighted_tokens),
            weighted_requests=float(weighted_requests),
        )

    async def reconcile(self, key: str, estimated_tokens: int, actual_tokens: int) -> None:
        """Adjust the token counter by the delta between estimate and real usage."""
        sha = await self._get_script_sha()
        now_ms = int(time.time() * 1000)
        delta = actual_tokens - estimated_tokens
        await self._redis.evalsha(
            sha,
            1,
            key,
            now_ms,
            self._window_s,
            self._rpm_limit,
            self._tpm_limit,
            0,
            "reconcile",
            delta,
        )
