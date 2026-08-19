"""Tests for the per-worker in-memory CircuitBreaker state machine."""

from fastapi_ctx_gateway.circuit_breaker import CircuitBreaker, CircuitState


def _clock(times: list[float]):
    it = iter(times)

    def _now() -> float:
        return next(it)

    return _now


def test_closed_allows_requests() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_success_keeps_it_closed_and_resets_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED  # two failures after reset, still under threshold


def test_trips_open_after_failure_threshold() -> None:
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout_s=30, now=_clock([0.0, 0.0, 0.0]))
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_transitions_to_half_open_after_reset_timeout() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=30, now=_clock([0.0, 0.0, 35.0]))
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.state == CircuitState.HALF_OPEN  # 35s later


def test_half_open_allows_only_bounded_trial_calls() -> None:
    breaker = CircuitBreaker(
        failure_threshold=1, reset_timeout_s=30, half_open_max_calls=1, now=_clock([0.0, 35.0])
    )
    breaker.record_failure()
    assert breaker.allow_request() is True  # the one trial call (also flips to half-open)
    assert breaker.allow_request() is False  # no more until it resolves


def test_half_open_success_closes_circuit() -> None:
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=30, now=_clock([0.0, 35.0]))
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_half_open_failure_reopens_circuit() -> None:
    breaker = CircuitBreaker(
        failure_threshold=1, reset_timeout_s=30, now=_clock([0.0, 35.0, 40.0, 40.0])
    )
    breaker.record_failure()
    breaker.allow_request()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
