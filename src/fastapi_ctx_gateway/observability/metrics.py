"""Prometheus counters for every fallback/degradation decision in the gateway.

Hand-instrumented rather than delegated to `prometheus-fastapi-instrumentator`
(used elsewhere for generic HTTP metrics only) — these are business-logic
events, not inferable from request/response shape.
"""

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter

__all__ = ["Metrics", "build_metrics"]


@dataclass
class Metrics:
    """One Counter per gateway-specific event, all bound to the same registry."""

    registry: CollectorRegistry
    cache_hit: Counter
    cache_miss: Counter
    prune_triggered: Counter
    rate_limit_rejected: Counter
    circuit_breaker_open: Counter
    vector_store_fail_open: Counter


def build_metrics(registry: CollectorRegistry | None = None) -> Metrics:
    """Build a fresh set of counters. Pass an isolated registry in tests."""
    registry = registry if registry is not None else CollectorRegistry()
    return Metrics(
        registry=registry,
        cache_hit=Counter("cache_hit_total", "Semantic cache hits", registry=registry),
        cache_miss=Counter(
            "cache_miss_total",
            "Semantic cache misses (including bypassed/fail-open)",
            registry=registry,
        ),
        prune_triggered=Counter(
            "prune_triggered_total",
            "Requests where pruning actually changed the contents sent upstream",
            registry=registry,
        ),
        rate_limit_rejected=Counter(
            "rate_limit_rejected_total",
            "Requests rejected by the rate limiter",
            registry=registry,
        ),
        circuit_breaker_open=Counter(
            "circuit_breaker_open_total",
            "Requests short-circuited because the Gemini circuit breaker was open",
            registry=registry,
        ),
        vector_store_fail_open=Counter(
            "vector_store_fail_open_total",
            "Cache lookups/stores that failed and fell open (treated as a miss)",
            registry=registry,
        ),
    )
