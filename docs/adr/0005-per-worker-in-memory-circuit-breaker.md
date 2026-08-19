# ADR-0005: Per-worker in-memory circuit breaker, not a Redis-synchronized one

## Context

The circuit breaker protects the gateway (and Gemini) from hammering a failing upstream. Its trip decision could be centralized across all workers via a synchronous Redis check on every request, giving cluster-wide consistency, or kept local to each worker process, in memory, with no I/O on the hot path.

## Decision

Per-worker, in-memory. State is best-effort propagated to Redis for cross-worker dashboards only — it is never read back to make the allow/deny decision.

## Consequences

A synchronous cross-worker check would require a Redis round trip on every single request specifically to protect against Redis/Gemini being unreliable — reintroducing, on the breaker's own hot path, the exact latency and failure-coupling cost the breaker exists to eliminate. The tradeoff: each worker trips independently, so a cluster of N workers takes up to N times the failure count to fully "notice" an outage cluster-wide, and workers can briefly disagree on state. This is judged acceptable — the breaker's job is protecting each worker's own request-serving capacity and the shared Gemini quota, not providing perfectly synchronized cluster observability (which the best-effort Redis propagation covers for dashboards, separately).
