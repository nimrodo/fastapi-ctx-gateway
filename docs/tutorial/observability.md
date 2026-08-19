# Observability

Every fallback and degradation decision in the gateway is instrumented — an invisible fallback isn't graceful, it's silent.

## Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

Domain-specific counters (hand-instrumented — see below for why):

| Counter | Fires when |
|---|---|
| `cache_hit_total` | A semantically similar cached response was found and replayed |
| `cache_miss_total` | No hit — includes genuine misses, ineligible requests, and fail-open |
| `prune_triggered_total` | A conversation actually exceeded its token budget and got pruned |
| `rate_limit_rejected_total` | A request was rejected by the rate limiter |
| `circuit_breaker_open_total` | A request was short-circuited by an open breaker |
| `vector_store_fail_open_total` | A cache lookup/store failed and fell open (a real backend failure, not a genuine miss) |

Generic HTTP metrics (request counts, status codes, latency histograms) come from [`prometheus-fastapi-instrumentator`](https://github.com/trallnag/prometheus-fastapi-instrumentator), served from the same endpoint. It's used for that baseline only — it doesn't reliably measure SSE stream duration (a known gap: ASGI middleware timing typically stops at "headers sent," not stream completion), so the latency-critical spans below are hand-instrumented instead.

## OpenTelemetry spans

Two spans map directly to the [two latency budgets](../index.md#two-latency-budgets):

- **`cache.hit_path`** — wraps embed + cache lookup + response synthesis on a cache hit.
- **`request.pre_proxy`** — wraps the rate-limit check, circuit-breaker check, and pruning on a miss, ending right before the first byte is proxied from Gemini.

They're sibling spans, not nested — a given request emits at most one of them. See [`hit_path_span`][fastapi_ctx_gateway.observability.tracing.hit_path_span] and [`pre_proxy_span`][fastapi_ctx_gateway.observability.tracing.pre_proxy_span]. Configure a real exporter destination via the standard `OTEL_*` environment variables in your deployment — the gateway doesn't hard-wire one.

## Reading the fail-open metric

`vector_store_fail_open_total` climbing is the one metric worth alerting on directly: it means the semantic cache is *degraded*, not just occasionally missing. A healthy gateway with the cache enabled should see this stay at (or very near) zero.
