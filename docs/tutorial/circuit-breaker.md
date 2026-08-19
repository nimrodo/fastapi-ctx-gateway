# Circuit breaker

If Gemini itself is failing, hammering it with every incoming request just makes things worse — for Gemini, and for the gateway's own capacity. The [`CircuitBreaker`][fastapi_ctx_gateway.circuit_breaker.CircuitBreaker] short-circuits requests during an outage instead.

## States

`CLOSED` → `OPEN` → `HALF_OPEN` → `CLOSED` or back to `OPEN`:

- **`CLOSED`** (normal) — requests proceed. Consecutive failures increment a counter; a success resets it to zero.
- **`OPEN`** — tripped after `GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD` consecutive failures. Requests are rejected immediately with `503`, without calling Gemini or even checking the cache.
- **`HALF_OPEN`** — after `GATEWAY_CIRCUIT_BREAKER_RESET_TIMEOUT_S`, a single trial request is allowed through. Success closes the circuit; failure reopens it immediately.

```bash
curl -i -X POST http://localhost:8000/v1/gemini-3.7-flash:streamGenerateContent \
  -H "x-gateway-api-key: my-gateway-key" -d '{"contents": [...]}'
# HTTP/1.1 503 Service Unavailable   (only while the breaker is open)
```

## Per-worker, in-memory — on purpose

The breaker holds no Redis round trip on its hot path. State is best-effort propagated to Redis for cross-worker dashboards, but the allow/deny decision is always local and O(1). See [ADR-0005](../adr/0005-per-worker-in-memory-circuit-breaker.md) for the reasoning — a synchronous cross-worker check would reintroduce, on the breaker's own path, the exact latency cost it exists to eliminate.

## Tuning

```bash
GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
GATEWAY_CIRCUIT_BREAKER_RESET_TIMEOUT_S=30.0
```

## Ordering

Checked right after the rate-limit check — an in-memory check is essentially free, so it happens before any pruning or embedding work that would otherwise be wasted on a request that's about to be short-circuited anyway.
