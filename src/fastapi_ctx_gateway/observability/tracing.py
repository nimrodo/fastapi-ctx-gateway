"""OTel span helpers for the two latency-budget boundaries.

`cache.hit_path` covers embed + cache lookup + response synthesis on a
cache hit (target: <=15-20ms). `request.pre_proxy` covers rate-limit +
breaker-check + prune, ending right before Gemini is called on a miss
(target: low single-digit ms, additive to Gemini's own latency). They're
sibling spans, not nested — a given request emits at most one of them
plus (on a miss) the proxy call itself.

Each helper accepts an optional `tracer`, defaulting to the process-wide
one resolved against whatever TracerProvider app.py installed at
startup. Accepting an override (rather than only ever reading the global)
is what makes these testable — OTel's global TracerProvider can only be
set once per process, so tests inject an isolated tracer/exporter
instead of fighting that global-state lock.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.trace import Tracer

__all__ = ["get_tracer", "hit_path_span", "pre_proxy_span"]


def get_tracer() -> Tracer:
    """Return the tracer resolved against the process-wide TracerProvider."""
    return trace.get_tracer("fastapi_ctx_gateway")


@contextmanager
def hit_path_span(tracer: Tracer | None = None) -> Iterator[None]:
    """Wrap embed + cache lookup + response synthesis on a cache-hit request."""
    with (tracer or get_tracer()).start_as_current_span("cache.hit_path"):
        yield


@contextmanager
def pre_proxy_span(tracer: Tracer | None = None) -> Iterator[None]:
    """Wrap rate-limit + breaker-check + prune, before the Gemini call on a miss."""
    with (tracer or get_tracer()).start_as_current_span("request.pre_proxy"):
        yield
