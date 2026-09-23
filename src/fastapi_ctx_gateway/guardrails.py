"""Request-side prompt-injection detection.

Request-side only: this looks for injection *patterns* in the raw text the
client sent, not for a jailbroken *response* — a cheap-ish first line of
defense, not a substitute for output-side moderation. `InjectionDetector`
is the seam (a stateless checker over `turns`/`system`, mirroring the
pruning/rate-limit shape) so it slots into the same pre-proxy sequencing
regardless of what backs it.

`detect()` must never raise: like `Provider.stream()` and the semantic
cache, a detector is an optional safety net, not a hard dependency for
serving traffic. `LocalClassifierInjectionDetector` fails open (returns
False, i.e. "no detection") on any inference failure or timeout.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from fastapi_ctx_gateway.schemas.neutral import Part, TextPart, Turn

__all__ = [
    "InjectionDetector",
    "LocalClassifierInjectionDetector",
    "PromptInjectionDetectedError",
    "placeholder_tokenize",
]

logger = logging.getLogger(__name__)


class PromptInjectionDetectedError(Exception):
    """Raised when mode=block and the detector matches.

    Bare marker, like `CircuitOpenError` — `detect()` returns only a bool,
    so there's no match detail to carry (or leak) even if we wanted to.
    """


class InjectionDetector(Protocol):
    """Checks a request's raw content for prompt-injection patterns.

    Implementations must never raise — a failure is the implementation's
    own responsibility to fail open (return False), the same contract
    `Provider.stream()` and the semantic cache follow.
    """

    async def detect(self, turns: list[Turn], system: list[Part] | None) -> bool:
        """Return True if any turn or system content looks like an injection attempt."""
        ...


def _collect_text(turns: list[Turn], system: list[Part] | None) -> str:
    parts = [part for turn in turns for part in turn.parts if isinstance(part, TextPart)]
    if system:
        parts += [part for part in system if isinstance(part, TextPart)]
    return "\n".join(part.text for part in parts)


class LocalClassifierInjectionDetector:
    """Runs a local ONNX text-classification model over the request's text.

    Never constructs its own session — built once in the app's lifespan
    and shared across requests, same as OnnxVectorizer. Inference is
    sync/CPU-bound, so it's always dispatched to a worker thread and
    bounded by `timeout_s`; any exception or timeout fails open (returns
    False) and reports through `on_fail_open`, mirroring SemanticCache's
    fail-open wrapper.
    """

    def __init__(
        self,
        session: Any,
        tokenize: Callable[[str], list[int]],
        threshold: float,
        timeout_s: float,
        on_fail_open: Callable[[], None] | None = None,
    ) -> None:
        """Wire in the ONNX session, tokenizer, decision threshold, and fail-open bound."""
        self._session = session
        self._tokenize = tokenize
        self._threshold = threshold
        self._timeout_s = timeout_s
        self._on_fail_open = on_fail_open

    async def detect(self, turns: list[Turn], system: list[Part] | None) -> bool:
        """Classify the request's text, off the event loop thread, bounded by a timeout."""
        text = _collect_text(turns, system)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._classify, text), timeout=self._timeout_s
            )
        except Exception:
            logger.warning("prompt-injection classifier failed; failing open", exc_info=True)
            if self._on_fail_open is not None:
                self._on_fail_open()
            return False

    def _classify(self, text: str) -> bool:
        """Run inference and threshold the softmax probability of the "injection" class.

        The model outputs two logits, [benign, injection] (see
        tests/fixtures/generate_tiny_classifier_model.py for the fixture
        shape; a real exported model follows the same two-logit contract).
        """
        token_ids = self._tokenize(text)
        input_array = np.array(token_ids, dtype=np.int64)
        input_name = self._session.get_inputs()[0].name
        (logits,) = self._session.run(None, {input_name: input_array})
        probabilities = _softmax(np.asarray(logits, dtype=np.float64))
        injection_probability = float(probabilities[1])
        return injection_probability >= self._threshold


def placeholder_tokenize(text: str) -> list[int]:
    """A dependency-light placeholder tokenizer: char code mod a fixed vocab size.

    Not semantically meaningful — mirrors cache/vectorizer.py's
    simple_char_code_tokenize. Swap in the real tokenizer that ships with
    whatever model prompt_injection_model_path points at (see
    scripts/download_model.py). Good enough to make the detection pipeline
    exercise-able end to end today.
    """
    return [ord(c) % 1000 for c in text] or [0]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)  # numerically stable
    exp = np.exp(shifted)
    result: np.ndarray = exp / exp.sum()
    return result
