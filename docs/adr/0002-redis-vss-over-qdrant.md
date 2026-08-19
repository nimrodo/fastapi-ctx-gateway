# ADR-0002: Redis + RediSearch/VSS for the semantic cache, not a dedicated vector database

## Context

The semantic cache needs a vector similarity index. Candidates considered: Redis with the RediSearch/VSS module (an index on an already-required Redis instance), or a dedicated vector database such as Qdrant (a purpose-built engine, but a second network hop and a second piece of infrastructure to operate).

## Decision

Use Redis + RediSearch/VSS. Redis is already a hard dependency for rate limiting, so this adds a search index on infrastructure already in the request path rather than introducing a second service and a second round trip.

## Consequences

Against a 15-20ms cache-hit budget, an additional network hop to a separate vector database on every request was judged too costly relative to the benefit. This does trade away some of what a dedicated vector engine offers at very large scale (recall tuning, advanced filtering, horizontal scaling independent of the KV workload). Revisit if the cached-entry corpus grows large enough, or recall quality demands, that a dedicated engine's capabilities clearly outweigh the added hop.
