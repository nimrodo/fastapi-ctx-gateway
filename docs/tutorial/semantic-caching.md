# Semantic caching

Near-duplicate prompts don't need to hit Gemini again. The semantic cache embeds the (pruned) conversation, looks it up against a Redis vector index, and — on a hit — replays the cached response without ever calling Gemini.

## Enabling it

The cache is **disabled by default**. Point it at an ONNX embedding model to turn it on:

```bash
GATEWAY_EMBEDDING_MODEL_PATH=/path/to/embedding.onnx
```

With no model configured (or a missing/unloadable file, or an unreachable Redis at startup), the gateway boots normally in "cache disabled" mode — every request is a guaranteed miss. The cache is a pure optimization; it can never be a reason the gateway fails to start or serve traffic. See [`SemanticCache`][fastapi_ctx_gateway.cache.SemanticCache] and [ADR-0003](../adr/0003-redisvl-semantic-cache-adoption.md).

!!! note
    A real production embedding model is a separate deploy-time asset (e.g. a quantized `all-MiniLM-L6-v2`). Until one is configured, the pipeline uses a placeholder tokenizer good enough to exercise the cache mechanics, not to produce meaningful embeddings.

## Eligibility

Not every request is cache-eligible — only ones whose output is meant to be near-deterministic:

- `generationConfig.tools` must be empty. A cached response can't reflect a different tool availability.
- `generationConfig.temperature` must be set and at or below `GATEWAY_CACHE_TEMPERATURE_THRESHOLD` (default `0.3`). An **unset** temperature is treated as "not explicitly low" — it bypasses the cache, it doesn't default into it.

```json
{"contents": [...], "generationConfig": {"temperature": 0.9}}
```

— always bypasses the cache; `X-Cache: MISS` on every call.

## Tuning

```bash
GATEWAY_CACHE_DISTANCE_THRESHOLD=0.10   # cosine distance; lower = stricter match required
GATEWAY_CACHE_TTL_S=3600                # cache entry lifetime
GATEWAY_CACHE_TEMPERATURE_THRESHOLD=0.3
GATEWAY_CACHE_LOOKUP_TIMEOUT_MS=50      # bounds the fail-open path
```

## Partitioning and failure

Every cache entry is tagged with `tenant_id` and `model`, and lookups always filter on both — there is no cross-tenant or cross-model cache hit, ever. Any lookup or store failure (timeout, connection error, anything) is caught and logged, never raised — see [Observability](observability.md) for the `vector_store_fail_open_total` counter that tracks exactly this.
