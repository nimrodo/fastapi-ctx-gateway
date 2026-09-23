# Security guarantees

The gateway's security posture against a **malicious-tenant** threat model: an
authenticated caller (a valid `x-gateway-api-key` holder) attempting to
reach another tenant's data, or to corrupt gateway-internal state, rather
than an external party without a key at all. Scoped to the gateway's own
code and operational posture — not to what a tenant's own downstream
application does with the responses it receives.

Vocabulary follows [`CONTEXT.md`](concepts.md): **Tenant boundary** (the
umbrella guarantee), **Partitioning** (the cache's enforcement mechanism),
**Content leak** (credential/content readable outside its request/response
path).

This doc was assembled from a
[wayfinder map](https://github.com/nimrodo/fastapi-ctx-gateway/issues/4)
that worked six decision tickets to ground; each guarantee below links back
to the ticket holding its full reasoning.

## Confirmed guarantees (no fix needed)

### Partitioning holds

The semantic cache's `{tenant_id, model}` tag filter is applied identically
on every lookup and store path (`SemanticCache.lookup`/`.store`), and
RedisVL passes that filter straight into the underlying vector query — not
a post-hoc re-rank. Proven empirically, not just by inspection:
`tests/integration/test_semantic_cache.py::test_different_tenant_is_a_miss`
stores under one tenant and asserts a lookup for the same prompt under
another tenant misses.

→ [Partitioning: confirm + regression-test the tenant/model cache filter](https://github.com/nimrodo/fastapi-ctx-gateway/issues/5)

### Relay fidelity holds

The gateway never parses or acts on model output as instructions. Every
provider's stream translator only converts upstream event *shape* into the
neutral schema — it never inspects or branches on output *content*
(confirmed: zero occurrences of `tool_call`/`function_call`/`exec`/`eval`/
`subprocess`/`os.system` anywhere in `src/`). Any future feature that does
inspect output content (gateway-side tool execution, content moderation,
output-triggered routing) must revisit this guarantee before shipping — it
would be the first thing to break it.

→ [Relay fidelity: confirm no model output is parsed or acted on](https://github.com/nimrodo/fastapi-ctx-gateway/issues/7)

### Pruning-budget accounting is immune to crafted-turn corruption

Neither a cross-tenant nor a same-tenant race is possible. Cross-tenant:
`TokenBudgetPruner.prune()` is a pure function of its arguments, no shared
state; the rate limiter's Redis key is `{tenant_api_key}:{model}`, so one
tenant's crafted input can corrupt only its own accounting, never another
tenant's. Same-tenant: the sliding-window Lua script
(`scripts/sliding_window.lua`) runs atomically and single-threaded in
Redis, and its increments (`HINCRBY`) are commutative, so concurrent
requests from the same tenant can't produce a lost update either.

→ [Crafted-turn attack on pruning-budget accounting](https://github.com/nimrodo/fastapi-ctx-gateway/issues/9)

### Semantic-cache logging paths are sound

`SemanticCache`'s two `logger.warning(..., exc_info=True)` sites can't leak
tenant content or Redis credentials given the vectorizer/client actually
wired in today: `OnnxVectorizer` does local ONNX inference (no HTTP
exception path), and every Redis client exception on the connect/read/write
paths interpolates only `host:port`, never a password or the raw
connection URL.

**Caveat**: if `OnnxVectorizer` is ever swapped for an HTTP-based
vectorizer (e.g. an OpenAI-embeddings vectorizer), re-run this analysis —
that would reintroduce a live network-exception path this finding doesn't
cover.

→ [Content leak: audit logging + error-relay paths](https://github.com/nimrodo/fastapi-ctx-gateway/issues/6)

## Confirmed gaps (fixes recommended)

### Canonicalization collision — same-tenant content leak

`canonicalize_turns` (`cache/serialize.py`) joins turn text with `\x00`/
`\x1e` as field/turn separators, and nothing rejects those literal bytes
from tenant-supplied text. Demonstrated, not hypothetical: a crafted
single-turn message can canonicalize identically to a genuine multi-turn
conversation, forcing a guaranteed cache hit (the current vectorizer is a
deterministic placeholder, so a collision means distance zero, not a
near-miss) that retrieves an unrelated prior conversation's cached
response under the same API key. This does **not** cross the tenant
boundary — the `{tenant_id, model}` tag filter still holds — but it is a
same-tenant content leak, notably worse for deployments where one API key
fronts many distinct end users. A lower-severity sibling of the same class
exists in `TokenBudgetPruner._fingerprint` (`pruning.py`), which uses the
identical `\x00`-join scheme for exact-duplicate detection; there the
consequence is wrongly-deduped turns, not cache/tenant correctness.

**Recommended fix**: switch both `canonicalize_turns` and
`TokenBudgetPruner._fingerprint` to a length-prefixed (netstring-style)
encoding per field — injective over the full `str` domain by construction,
no product decision needed about whether NUL/RS are ever legitimate chat
content, and non-breaking against the existing test suite. Reject-at-
validation is a reasonable later defense-in-depth addition, not a
substitute for fixing the encoding itself.

→ [Canonicalization collision: control bytes in turn text](https://github.com/nimrodo/fastapi-ctx-gateway/issues/8)
(findings + reproduction: `docs/research/issue-8-canonicalization-collision.md`
on branch `research/canonicalization-collision`)

### OpenAI error-relay — upstream-credential content leak

`parse_error_message` (`providers/sse.py`) relays the upstream error body's
`message` field straight to the client. Gemini's and Anthropic's
documented error bodies carry no content or key fragments. OpenAI's does:
a malformed-key 401 has the shape `"Incorrect API key provided:
<masked-key>."`, where `<masked-key>` is a partially-masked fragment of
the *actual key sent on the request* — this gateway's own
`GATEWAY_OPENAI_API_KEY`. A misconfigured, rotated, or truncated upstream
key would relay a fragment of the gateway's real OpenAI key to whichever
tenant happened to trigger the 401.

**Recommended fix**: before relaying `error.message`, strip any literal
occurrence of the gateway's own configured upstream key, or regex-scrub
known key-shape prefixes (`sk-...`, `sk-ant-...`, `AIza...`) as a
provider-agnostic backstop. Leave Gemini/Anthropic messages untruncated
otherwise — no content or key fragments in their documented bodies today,
so blanket truncation would only cost debuggability.

→ [Content leak: audit logging + error-relay paths](https://github.com/nimrodo/fastapi-ctx-gateway/issues/6)
(findings: `docs/research/issue-6-content-leak-audit.md` on branch
`research/content-leak-audit`)

### Auth: timing side-channel and no failed-auth backstop

`resolve_tenant`'s dict lookup (`tenant_api_keys.get(api_key)`) isn't
constant-time, and there is no rate limiting on failed auth attempts — the
existing `RateLimiter` only applies *after* successful auth. Because this
is a self-hosted/library gateway (ADR-0008), the maintainer doesn't control
how a given deployment is exposed, so both are assessed against a
worst-case (potentially internet-facing) assumption rather than an
unstated trust boundary.

**Recommended fixes**:

1. Constant-time key comparison — swap the plain dict lookup for
   `hmac.compare_digest` (or an equivalent constant-time approach).
2. Pre-auth failure rate limiting — add a backstop against unlimited 401
   attempts against a low-entropy or leaked key. Mechanism (per-source-IP
   vs. per-attempted-key) and thresholds are left as an implementation
   choice; this only locks that such a backstop should exist.

→ [Auth/tenant-key hygiene: dict lookup, plaintext keys, no rate limiting](https://github.com/nimrodo/fastapi-ctx-gateway/issues/10)

## Out of scope

- Tenant-facing security guidance — a docs page telling tenants what
  *they're* responsible for downstream of this gateway. Different
  audience/deliverable; left for a future effort if wanted.
- Implementation of the fixes recommended above. This doc is the spec to
  hand off; the wayfinder map that produced it
  ([issue #4](https://github.com/nimrodo/fastapi-ctx-gateway/issues/4))
  deliberately stopped short of building them.
