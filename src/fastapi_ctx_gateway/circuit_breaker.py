"""Per-worker in-memory circuit breaker for calls to one upstream provider.

Deliberately in-memory, not Redis-backed: a synchronous cross-worker
check would reintroduce the exact round-trip cost this exists to avoid
during an incident. State is best-effort propagated to Redis elsewhere
(dashboards only) — never read back for the trip decision. The gateway
builds one instance per registered provider (see app.py) so an outage on
one upstream never trips the breaker for an unrelated one.
"""

import time
from collections.abc import Callable
from enum import StrEnum

__all__ = ["CircuitBreaker", "CircuitOpenError", "CircuitState"]


class CircuitState(StrEnum):
    """The breaker's state machine states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a request is rejected because the breaker is open."""


class CircuitBreaker:
    """CLOSED -> OPEN after N consecutive failures -> HALF_OPEN trial -> CLOSED or OPEN again."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure trip thresholds; `now` is injectable for deterministic tests."""
        self._failure_threshold = failure_threshold
        self._reset_timeout_s = reset_timeout_s
        self._half_open_max_calls = half_open_max_calls
        self._now = now
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Current state, lazily transitioning OPEN -> HALF_OPEN once the timeout elapses."""
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            if self._now() - self._opened_at >= self._reset_timeout_s:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """O(1), no I/O: whether a request should be allowed to proceed to Gemini."""
        current = self.state
        if current is CircuitState.CLOSED:
            return True
        if current is CircuitState.OPEN:
            return False
        if self._half_open_calls < self._half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        """A successful call closes the circuit and resets the failure count."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None
        self._half_open_calls = 0

    def record_failure(self) -> None:
        """A failed trial reopens immediately; otherwise trip once the threshold is hit."""
        if self.state is CircuitState.HALF_OPEN:
            self._trip()
            return
        self._failure_count += 1
        if self._failure_count >= self._failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._now()
        self._failure_count = self._failure_threshold
        self._half_open_calls = 0
