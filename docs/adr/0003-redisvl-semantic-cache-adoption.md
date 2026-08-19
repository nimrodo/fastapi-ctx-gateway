# ADR-0003: Adopt RedisVL's SemanticCache rather than hand-rolled FT.SEARCH

## Context

Given ADR-0002 (Redis + RediSearch/VSS), the cache layer still needs: index management, a combined KNN + metadata-filter query, TTL handling, and an embedding-vectorizer abstraction. This can be hand-rolled directly against `redis-py` and `FT.SEARCH`, or built on RedisVL, a library purpose-built for exactly this pattern (one `Hash`/`JSON` entry per cache row holding prompt, embedding, response, and metadata; one `FT.SEARCH` call does KNN plus a `TAG`/`NUMERIC` pre-filter in the same pass).

## Decision

Adopt RedisVL's `SemanticCache` (`redisvl.extensions.cache.llm`), with a custom `BaseVectorizer` subclass (`OnnxVectorizer`) wrapping the local ONNX embedding model rather than RedisVL's default hosted vectorizer.

## Consequences

This satisfies tenant/model partitioning (`filterable_fields`, ADR-driven by the multi-tenancy requirement), TTL, and async I/O (`acheck`/`astore`) with meaningfully less custom code than hand-rolling index schema and query construction. The tradeoff: RedisVL's `SemanticCache.__init__` connects to Redis *eagerly at construction* (even with `overwrite=True`) — there is no way to construct one against an unreachable Redis. The gateway's startup code (`app.py::_build_semantic_cache`) guards this explicitly: a failed construction disables the cache rather than crashing the app, consistent with the fail-open contract applying at boot time, not just at request time (see `CONTEXT.md`).
