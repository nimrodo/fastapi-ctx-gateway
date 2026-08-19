# ADR-0004: Sliding-window-counter rate limiting, not an exact sorted-set log

## Context

Token-aware rate limiting (RPM/TPM per `{tenant}:{model}`) needs a windowing algorithm. An exact sliding-window log (a Redis sorted set per key, trimmed on each request via `ZREMRANGEBYSCORE`) gives precise accounting but costs O(log n) memory/CPU per request at high QPS. A sliding-window-*counter* approximation (two fixed-window counters, weighted by how far into the current window the request lands) is O(1) per request but tolerates a bounded error at window boundaries (on the order of 1-2%).

## Decision

Use the sliding-window-counter approximation, implemented as a single atomic Lua script (`scripts/sliding_window.lua`, loaded once via `SCRIPT LOAD`, invoked by `EVALSHA` — one round trip per check or reconcile call).

## Consequences

Bounded imprecision at window edges is an accepted tradeoff against a 15-20ms overall latency budget, where the O(log n) sorted-set approach would add measurable per-request cost with no material benefit to the gateway's actual admission-control goal (protecting Gemini quota and gateway capacity, not billing-grade accounting). The Lua script keeps the whole check-and-consume operation atomic in one round trip rather than a check-then-increment race across two calls.
