# Research: LangChain/LangGraph multimodal message-content conventions

Issue: [nimrodo/fastapi-ctx-gateway#42](https://github.com/nimrodo/fastapi-ctx-gateway/issues/42)
(parent map: #41, blocking implementation ticket: #43)

Scope: what `_to_agent_messages()` in `src/fastapi_ctx_gateway/providers/agent.py` should
produce for a `Turn` that mixes `TextPart`/`BinaryPart`, so it round-trips into both a plain
LangChain `Runnable` and a LangGraph `CompiledStateGraph`'s `messages` state key.

Date of research: 2026-09-24. LangChain 1.0 / LangGraph 1.0 reached GA on **2025-10-22**
([LangChain blog](https://www.langchain.com/blog/langchain-langgraph-1dot0),
[Microsoft Community Hub recap](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/langchain-v1-is-now-generally-available/4462159)),
so "current" below means the v1 line, not the older v0.1–v0.3 series.

---

## 1. Current representation of multimodal `HumanMessage.content`

LangChain message `content` has always been either a plain `str` or a `list` of
"content blocks" (dicts with a `type` discriminator). What changed in 1.0 is that
`langchain_core` now defines its **own vendor-neutral block shapes** in
`langchain_core.messages.content` (also re-exported/described in some docs/search
results as `content_blocks` — the current source module on `master` is
`langchain_core/messages/content.py`), instead of every integration inventing its own.
Source: <https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/content.py>.

There are now **two generations** of vendor-neutral shape in circulation. Both are
real and both are still accepted; they differ in field names.

### 1a. The "v1" native shape (current, `langchain_core` ≥ 1.0)

`ImageContentBlock` (and the structurally-identical `AudioContentBlock`,
`VideoContentBlock`, `FileContentBlock`) is a `TypedDict`:

```python
class ImageContentBlock(TypedDict):
    type: Literal["image"]  # discriminator
    id: NotRequired[str]  # auto lc_-prefixed UUID4 if omitted
    url: NotRequired[str]  # by-reference
    base64: NotRequired[str]  # inline, requires mime_type
    file_id: NotRequired[str]  # provider-managed file reference
    mime_type: NotRequired[str]
    index: NotRequired[int | str]  # streaming position
    extras: NotRequired[dict[str, Any]]  # provider-specific passthrough
```

Source: <https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/content.py>
(confirmed via WebFetch of the raw GitHub source) and the reference pages
<https://reference.langchain.com/python/langchain-core/messages/content/ImageContentBlock>
and <https://reference.langchain.com/python/langchain-core/messages/content>.

Construction is via a factory, not the bare dict, though the dict is what actually
ends up in `.content`:

```python
from langchain_core.messages.content import create_image_block

create_image_block(url="https://…/img.png", mime_type="image/png")
create_image_block(base64="<b64>", mime_type="image/png")
```

`create_image_block(url=…, base64=…, file_id=…, mime_type=…, id=…, index=…, **kwargs)`
validates that at least one of `url`/`base64`/`file_id` is given and raises
`ValueError` if `base64` is given without `mime_type`.
Source: <https://reference.langchain.com/python/langchain-core/messages/content/create_image_block>.

There is **no separate `Base64ContentBlock`/`URLContentBlock` base type** — `url`,
`base64`, and `file_id` are just optional sibling keys on the *same* `TypedDict`, and
the consumer picks whichever key is present. This is confirmed by the source file
directly, not just docs.

A message's `.content_blocks` property (added in 1.0) lazily parses whatever is in
`.content` — legacy string, legacy provider-native dicts, or these new blocks — into
this unified, typed representation. `.content` itself is unchanged and still accepts
"strings and lists of untyped objects (dictionaries)".
Source: <https://docs.langchain.com/oss/python/langchain/messages> (WebFetch, section
"Content Format Flexibility" / `"loosely-typed"` content quote), and the announcement
blog <https://www.langchain.com/blog/standard-message-content>.

### 1b. The "v0.3-era" cross-provider shape (older, still accepted as input)

Before the 1.0 `content.py` blocks, LangChain (from ~mid-2024, the "multimodality"
push in 0.3) already had a *different* vendor-neutral shape, keyed by `source_type`
rather than having `url`/`base64` as sibling keys directly:

```python
# inline base64
{"type": "image", "source_type": "base64", "data": "<b64>", "mime_type": "image/png"}
# by-reference URL
{"type": "image", "source_type": "url", "url": "https://…/img.png", "mime_type": "image/png"}
```

This is confirmed by `langchain_core/messages/block_translators/langchain_v0.py`,
whose entire job is to translate *this* legacy shape into the new `content.py` shapes
so `.content_blocks` can normalize both generations transparently:

> "Old v0 Block Shapes — Base64 Image (v0): `{"type": "image", "source_type": "base64", "data": "<base64_string>", "mime_type": "image/png"}`. URL-Referenced Image (v0): `{"type": "image", "source_type": "url", "url": "https://example.com/image.png", "mime_type": "image/png"}`"

Source: <https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/block_translators/langchain_v0.py>
(WebFetch of raw GitHub source).

**Both shapes are live today.** `content.py`/`content_blocks` normalizes the v0
`source_type` shape into the v1 shape on read, and the v0 shape remains
what most of the multimodality how-to material in LangChain's docs still shows for
*constructing* input messages (it predates and is independent of the 1.0 output
`.content_blocks` parsing work). The `.content` field itself has always accepted
provider-native blocks too — e.g. OpenAI's own `{"type": "image_url", "image_url":
{"url": …}}` — and that continues to work unchanged; `content.py`'s
`block_translators/openai.py`/`anthropic.py`/`google_genai.py` submodules exist
specifically to translate each vendor's own wire-format blocks into/out of the
unified v1 shape. Source: <https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/block_translators/openai.py>,
<https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/block_translators/anthropic.py>.

### A known rough edge in the new normalization

[langchain-ai/langchain#40424](https://github.com/langchain-ai/langchain/issues/40424)
(open as of this research): accessing `.content_blocks` on a **legacy v0 block that
carries an `id` field** raises `TypeError: got multiple values for keyword argument
'id'`, because `id` isn't in the `known_keys` list for any of the seven
v0 image/audio/file × url/base64/text combinations, so it lands in `extras` *and*
gets passed positionally, double-supplying the constructor's `id` kwarg. Relevant if
we ever set an explicit `id` on emitted blocks — safest to omit `id` and let
LangChain auto-generate one.

---

## 2. Does the plain-`Runnable` path differ from the LangGraph `CompiledStateGraph` path?

**No — confirmed, not just assumed.** LangGraph's `MessagesState`/`add_messages`
reducer stores and forwards the *same* `BaseMessage` (or message-dict) objects a
chat model would accept directly; a `HumanMessage.content` list containing a mix of
`{"type": "text", ...}` and `{"type": "image_url", ...}`/`{"type": "image", ...}`
blocks is exactly what's shown in LangGraph's own docs example for a graph's message
state:

```python
HumanMessage(
    content=[
        {"type": "text", "text": "Here's an image:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,1234"}},
    ],
)
```

Source: <https://docs.langchain.com/oss/python/langgraph/use-graph-api> (via WebSearch
excerpt) and <https://reference.langchain.com/python/langgraph/graph/message/add_messages>.

One LangGraph-specific nuance worth flagging: `add_messages` accepts an optional
`format` parameter (e.g. `"langchain-openai"`) that reformats messages into the shape
a specific integration expects when messages are *read back out* of state — this is
an opt-in convenience, not something the gateway needs, since `AgentProvider` writes
directly to `{"messages": messages}` on `.astream()` input and doesn't rely on a
custom reducer format. No evidence this affects what shape of content block can be
*written into* the list.

Net: whatever content-block shape works as a chat model's/Runnable's direct input
also works as an entry in the list handed to `CompiledStateGraph.astream({"messages":
messages}, ...)` — there's one message representation, not two.

---

## 3. Known gaps / footguns by integration

Findings below come from reading `langchain_core`'s `block_translators/*.py` modules
directly (they define, per-vendor, which block shapes each integration's wire format
supports translating to/from) plus the parent GitHub issues found.

| Integration | Source | Gaps / footguns |
|---|---|---|
| **OpenAI** (`langchain-openai`, Chat Completions) | [`block_translators/openai.py`](https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/block_translators/openai.py) | Images: URL **and** base64 both supported, converted to `image_url` with a `data:` URI for base64. **Audio: base64 only** — no URL or `file_id` audio support in Chat Completions. **File URLs are rejected by Chat Completions** — the translator raises `ValueError("OpenAI Chat Completions does not support file URLs")`; file URLs only work through the separate Responses API path. **No video content-block handling at all.** Base64 file uploads without an explicit filename fall back to a placeholder (`LC_AUTOGENERATED`), which the code itself warns "OpenAI may require a filename for file uploads." |
| **Anthropic** (`langchain-anthropic`) | [`block_translators/anthropic.py`](https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/block_translators/anthropic.py) | Images and documents (files) both support base64, URL, **and** file-id (Anthropic Files API) sources on the way in/out. No explicit audio/video block translation — anything Anthropic itself doesn't emit in one of the recognized shapes falls back to a `"non_standard"` passthrough wrapper rather than erroring, which can silently drop structure if a required field (e.g. `data` on a base64 source) is missing. |
| **Google Gemini** (`langchain-google-genai`) | [`block_translators/google_genai.py`](https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/core/langchain_core/messages/block_translators/google_genai.py) | `file_data` (URI-based, i.e. Gemini's `fileUri`) maps to a **url**-sourced file block — by-reference works. Inline images/PDFs map to base64 blocks; MIME-type sniffing for images depends on the optional `filetype` package being installed, otherwise base64 image blocks can end up with no explicit `mime_type`. Audio support reads raw bytes out of `additional_kwargs`, defaulting to `audio/wav` if no type is given — narrower/more implicit than OpenAI/Anthropic's explicit MIME handling. Only `txt`/`pdf` get first-class "plain text document" handling; other document MIME types fall through to non-standard blocks. |

None of the three translator modules document an *inbound* (construction-time)
restriction that would stop us from emitting a `url`-sourced or `base64`-sourced
`image`/`file` block for any of the three vendors when we don't know in advance
which chat-model class is wrapped — the "unsupported" cases above are about a
specific *modality* (e.g. OpenAI audio-by-URL, OpenAI file-by-URL on Chat
Completions) that this gateway's `BinaryPart` doesn't yet distinguish by content
kind anyway (it's mime-type-driven, not modality-typed).

`astream`/`ainvoke` acceptance: nothing found in any of the three translator sources
or in LangChain's messages docs suggests streaming vs. non-streaming calls accept
different input-content shapes — the content-block format is a property of the
*message*, not of which invocation method consumes it. No evidence of an
`astream`-specific restriction was found; this should be spot-checked empirically in
#43 against whichever concrete `langchain-openai`/`langchain-anthropic` version is
pinned, since it wasn't independently confirmed against a live call in this research
pass.

Other adjacent open issues surfaced during this research (not blocking, but
worth being aware of if #43 also touches token counting/cost estimation for
multimodal agent requests):
[langchain-ai/langchain#40741](https://github.com/langchain-ai/langchain/issues/40741) /
[#40731](https://github.com/langchain-ai/langchain/issues/40731) /
[#40744](https://github.com/langchain-ai/langchain/issues/40744) — `count_tokens_approximately`
counts base64 image payloads as literal characters, over-counting tokens by roughly
1000x for the same image, on OpenAI `input_image` blocks.

---

## 4. Recommendation for `_to_agent_messages`

### What `content` should become

Per LangChain's own guidance (`.content` accepts "strings and lists of untyped
objects" — <https://docs.langchain.com/oss/python/langchain/messages>) and standard
LangChain idiom, `_to_agent_messages` should emit:

- **`str`** when a `Turn`'s parts are all `TextPart` (today's behavior, unchanged —
  this keeps every existing text-only caller, including any `Runnable` built from a
  plain `ChatPromptTemplate` expecting a bare string, working exactly as before).
- **`list[dict]`** (content blocks) when a `Turn` has at least one `BinaryPart`,
  interleaving `{"type": "text", "text": ...}` blocks for each `TextPart` and one
  binary block per `BinaryPart`, in original part order.

This mirrors LangChain's own dual acceptance and avoids a breaking change to every
text-only integration test/example in the repo and its docs
(`docs/tutorial/agent-provider.md`, `examples/`).

### Which block shape for `BinaryPart`

Two live options, both real and both accepted as `HumanMessage.content` input today:

**Option A — v0.3-era cross-vendor shape (`source_type`-keyed).** This is the shape
LangChain's own multimodality docs have shown since before 1.0, is transparently
normalized by `content.py`'s `langchain_v0.py` translator, and — critically — is the
shape most existing chat-model integrations have supported the *longest*, so it's
the safer bet for an `AgentProvider` that wraps an arbitrary, version-unpinned,
duck-typed object we don't control:

```python
# BinaryPart(data=...) -> inline
{"type": part.kind, "source_type": "base64", "data": part.data, "mime_type": part.mime_type}
# BinaryPart(uri=...) -> by-reference
{"type": part.kind, "source_type": "url", "url": part.uri, "mime_type": part.mime_type}
```

**Option B — v1-native shape (`content.py`'s `create_image_block`/`create_file_block`
family).** This is the forward-looking, officially-current shape
(<https://reference.langchain.com/python/langchain-core/messages/content>), simpler
(no `source_type` indirection), and is what `.content_blocks` normalizes *to*, not
*from*:

```python
# inline
{"type": part.kind, "base64": part.data, "mime_type": part.mime_type}
# by-reference
{"type": part.kind, "url": part.uri, "mime_type": part.mime_type}
```

**Recommendation: emit Option A (`source_type`-keyed) as the default translation**,
because:

1. `AgentProvider` explicitly does **not** know which chat-model integration (if any)
   sits behind the wrapped `Runnable`/`CompiledStateGraph` — it's duck-typed by
   design (see the module docstring: "no hard import of `langchain_core`/`langgraph`
   types"). The v0.3 `source_type` shape has the longest support tail across
   `langchain-openai`/`langchain-anthropic`/`langchain-google-genai` and is exactly
   what `content.py`'s own `langchain_v0.py` translator exists to keep working
   indefinitely — LangChain treats it as a first-class input shape, not a deprecated
   one, despite naming it "v0" internally.
2. It requires no `langchain_core` import in `agent.py` at all (consistent with the
   existing "duck-typed, no hard import" design constraint) — it's just a plain dict
   literal, same as today's `(role, content)` tuple approach.
3. Should a wrapped object be running an older `langchain_core` (< 1.0, pre-GA
   2025-10-22) that has no `content_blocks`/`content.py` module at all, the
   `source_type` shape is the one that predates and is independent of the 1.0 work,
   so it degrades gracefully rather than depending on 1.0-only normalization code
   being present.

A `mime_type`-to-`type` mapping is needed either way (`image/*` → `"image"`,
`audio/*` → `"audio"`, `application/pdf`/others → `"file"`), mirroring the same
kind of per-vendor branching the repo's `gemini.py`/`openai.py`/`anthropic.py`
translators already do off `BinaryPart.mime_type`.

If, later, telemetry or a specific pinned integration shows the v0 shape being
silently dropped or mis-normalized by a modern `langchain_core`, Option B is a
drop-in swap for the same call site — it's a strict subset of information (no
`source_type` field), so both can be produced from the same internal
`_binary_part_to_block(part)` helper with a single conditional if we ever want to
support both.

### Summary answer to Q4

Replace `_turn_text(parts) -> str` (which raises on `BinaryPart`) with a function
that returns `str | list[dict]`: `str` for text-only turns, `list[dict]` — text
blocks plus one `source_type`-keyed binary block per `BinaryPart` — otherwise. Both
a plain `Runnable`/chat model and a LangGraph graph's `messages` state accept this
uniformly (§2), so no special-casing is needed between the two `AgentProvider`
streaming paths (`_stream_text` vs `_stream_langgraph_events`) on the way in — only
`_to_agent_messages` itself needs to change.

---

## Gaps to flag to #43 (implementation ticket)

1. **No independent live-call confirmation.** Everything above is from reading
   `langchain_core`'s source/docs, not from actually invoking `ChatOpenAI`/
   `ChatAnthropic`/`ChatGoogleGenerativeAI` with a `source_type`-keyed block through
   `.astream()`. #43 should add a smoke/integration test against whichever
   `langchain-core`/`langchain-openai`/etc. versions the repo pins (or its own
   in-repo `examples/`) before shipping.
2. **`id` field footgun** ([#40424](https://github.com/langchain-ai/langchain/issues/40424)):
   don't set an explicit `id` on emitted blocks; let LangChain auto-generate one, or
   pin/test against a `langchain-core` version with that issue fixed.
3. **Per-vendor modality gaps** (§3): if `BinaryPart.mime_type` indicates audio and
   the wrapped agent turns out to be backed by `ChatOpenAI` on the Chat Completions
   API, only base64 audio works — a URL-referenced `BinaryPart` for audio would
   silently misbehave or error depending on how the underlying integration handles
   an unrecognized block. Since `AgentProvider` can't introspect which integration is
   wrapped, this is a documented limitation, not something `_to_agent_messages` can
   fix generically — worth a docstring note in the eventual `agent.py` diff, similar
   to the existing `_turn_text` docstring's issue-number backreference pattern.
4. **File uploads without a filename**: for `application/*`/file-typed `BinaryPart`s,
   OpenAI's translator falls back to a placeholder filename and warns it may not be
   accepted — `BinaryPart` has no filename field today; worth deciding in #43 whether
   to thread one through or accept the placeholder behavior.
5. **Token-estimation blind spot** (adjacent, not blocking): if this gateway's own
   token/cost estimation ever runs LangChain's `count_tokens_approximately` over
   agent-provider requests, base64 multimodal payloads are known to be wildly
   over-counted upstream ([#40741](https://github.com/langchain-ai/langchain/issues/40741),
   [#40731](https://github.com/langchain-ai/langchain/issues/40731),
   [#40744](https://github.com/langchain-ai/langchain/issues/40744)) — not relevant to
   #43's scope today since `AgentProvider` doesn't do token estimation, but worth a
   note if that ever changes.
6. **`langchain_core.messages.content` vs "`content_blocks`" naming**: some
   community/docs material (including third-party summaries surfaced during this
   research) refers to the module informally as `content_blocks`; the actual current
   module path on `master` is `langchain_core.messages.content`, confirmed by
   fetching the raw source directly — use that import path if #43 ever needs to
   type-check against the real classes rather than emitting plain dicts.
