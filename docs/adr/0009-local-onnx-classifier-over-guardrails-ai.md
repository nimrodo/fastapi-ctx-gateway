# ADR-0009: A local ONNX classifier, not `guardrails-ai`, backs prompt-injection detection

## Context

Issue #21 asked to replace #18's `HeuristicInjectionDetector` (plain regex over three known phrase categories — trivially bypassed by paraphrasing or light obfuscation) with a real ML classifier, and named `guardrails-ai`'s Guardrails Hub as the backend: "a Guardrails Hub prompt-injection validator (local classifier model, e.g. deberta-based) — no per-request network/LLM call, low latency."

Research done before implementation found that premise doesn't hold:

- The Hub validator literally named for this (`hub://guardrails/detect_prompt_injection`) uses the Rebuff library and makes per-request network calls to an LLM and a Pinecone index — the opposite of "no per-request network/LLM call." Its repo is archived (last activity Feb 2024).
- The Hub's actual local, no-network validator (`detect_jailbreak`) wraps a BERT (not DeBERTa) classifier scoped specifically to jailbreak-style phrasing ("you are DAN," roleplay-to-bypass-restrictions) — narrower than general prompt injection, and close enough to the old regex heuristic's own `_ROLEPLAY_JAILBREAK_PATTERNS` category that it wouldn't meaningfully improve on the instruction-override and fake-delimiter categories the heuristic also covered.
- No Guardrails Hub validator wraps a general-purpose, local, DeBERTa-based prompt-injection model. That model (`protectai/deberta-v3-base-prompt-injection-v2`) is real and does run fully locally, but only via loading it directly — not through `guardrails-ai`.
- Separately, Guardrails Hub's `guardrails hub install` CLI and private registry were fully shut down (Aug 25, 2026); the current install path for any Hub validator is a plain public-PyPI package, with no account/token required — this part is now simpler than the issue assumed, but doesn't change the coverage gap above.

Given this, adopting `guardrails-ai` would mean shipping a narrower detector (jailbreak phrasing only) than either the old heuristic's three categories or the issue's stated intent, coupled to a third-party library's own API and a Hub ecosystem that just went through a breaking migration.

## Decision

Load the classifier model directly via `onnxruntime`, not through `guardrails-ai`. `LocalClassifierInjectionDetector` (`guardrails.py`) wraps an injected ONNX `InferenceSession` + tokenizer function + threshold, exactly mirroring `OnnxVectorizer`'s existing shape for the semantic cache's embedding model:

- No new dependency extra: `onnxruntime` and `tokenizers` are already core dependencies (added for the embedding cache's own local-model story), so this backend needs nothing new installed.
- `Settings.prompt_injection_model_path: Path | None` gates the backend, the same pattern as `embedding_model_path` — point it at an exported ONNX model (e.g. `protectai/deberta-v3-base-prompt-injection-v2`, exported once, offline, via `optimum`) to enable it.
- `Settings.prompt_injection_backend: Literal["local_classifier"] | None` stays as an explicit selector (rather than inferring "on" purely from the model path being set) so a structurally different future backend — one that isn't "point at a local ONNX file," e.g. an LLM self-check à la `nemo-guardrails` — has a place to plug in without a schema break.
- Inference is sync/CPU-bound; `detect()` dispatches it via `asyncio.to_thread`, bounded by `prompt_injection_timeout_ms`, and fails open (treats a timeout or exception as "no match," never a request failure) — the same contract `SemanticCache.lookup()`/`store()` already follow, reported through a dedicated `prompt_injection_fail_open_total` counter mirroring `vector_store_fail_open_total`.

`guardrails-ai` remains a candidate worth revisiting if a future need (e.g. layered validators, a hosted moderation call) outgrows what a single local classifier can do — this decision is about what phase 1 needs today, not a rejection of the library in general.

## Consequences

- **The issue's literal "guardrails-ai" framing doesn't match the implementation**, and issue #21 is being corrected to reflect this rather than left inconsistent with the code.
- **One fewer runtime dependency than the issue implied** — no `guardrails-ai` package, no Hub account/token flow, no exposure to that library's post-migration API churn.
- **The detector's threat coverage is whatever the deployed model provides**, not fixed by a Hub validator's own scope — a general prompt-injection model (not just jailbreak phrasing) is a config choice, not a library constraint.
- **Model provisioning is a manual, offline step** (`scripts/download_model.py`, extended to also export this model), identical in shape to the embedding cache's still-unfinished story — no model ships in the repo or downloads automatically at boot or request time.
- **No fallback to the old regex heuristic remains** (per #21's non-goals) — the only rollback if a deployed model misbehaves in production is `prompt_injection_mode: off`, accepted because `flag`/`block` currently only observe (never reject), so the blast radius of a bad model is a wrong log line and metric, not wrongly blocked traffic.
