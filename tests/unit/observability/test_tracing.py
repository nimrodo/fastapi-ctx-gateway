"""Tests for the hit-path / pre-proxy OTel span helpers.

Each test builds its own isolated TracerProvider/exporter and injects
the resulting tracer directly, rather than touching the process-wide
global TracerProvider — that global can only be set once per process,
so mutating it per-test would leak across the test session.
"""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fastapi_ctx_gateway.observability.tracing import hit_path_span, pre_proxy_span


def _tracer_with_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def test_hit_path_span_is_named_correctly() -> None:
    tracer, exporter = _tracer_with_exporter()
    with hit_path_span(tracer):
        pass
    names = [span.name for span in exporter.get_finished_spans()]
    assert "cache.hit_path" in names


def test_pre_proxy_span_is_named_correctly() -> None:
    tracer, exporter = _tracer_with_exporter()
    with pre_proxy_span(tracer):
        pass
    names = [span.name for span in exporter.get_finished_spans()]
    assert "request.pre_proxy" in names


def test_spans_are_siblings_not_nested() -> None:
    tracer, exporter = _tracer_with_exporter()
    with pre_proxy_span(tracer):
        pass
    with hit_path_span(tracer):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    pre_proxy = next(s for s in spans if s.name == "request.pre_proxy")
    hit_path = next(s for s in spans if s.name == "cache.hit_path")
    # Two sibling spans, not nested — pre_proxy's interval doesn't overlap hit_path's.
    assert pre_proxy.end_time <= hit_path.start_time
    assert pre_proxy.parent is None
    assert hit_path.parent is None
