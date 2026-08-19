"""Tests for the gateway's Prometheus counters."""

from prometheus_client import CollectorRegistry

from fastapi_ctx_gateway.observability.metrics import build_metrics


def _value(counter) -> float:
    return counter.collect()[0].samples[0].value


def test_each_counter_starts_at_zero() -> None:
    metrics = build_metrics(CollectorRegistry())
    assert _value(metrics.cache_hit) == 0
    assert _value(metrics.cache_miss) == 0
    assert _value(metrics.prune_triggered) == 0
    assert _value(metrics.rate_limit_rejected) == 0
    assert _value(metrics.circuit_breaker_open) == 0
    assert _value(metrics.vector_store_fail_open) == 0


def test_counters_increment_independently() -> None:
    metrics = build_metrics(CollectorRegistry())
    metrics.cache_hit.inc()
    metrics.cache_hit.inc()
    metrics.cache_miss.inc()
    assert _value(metrics.cache_hit) == 2
    assert _value(metrics.cache_miss) == 1
    assert _value(metrics.prune_triggered) == 0


def test_fresh_registry_per_call_avoids_duplicate_registration() -> None:
    # build_metrics(None) creates its own isolated registry each time -
    # this must not raise (a shared default registry would on 2nd call).
    build_metrics()
    build_metrics()
