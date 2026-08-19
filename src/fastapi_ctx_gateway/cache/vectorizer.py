"""A local ONNX embedding vectorizer conforming to RedisVL's BaseVectorizer interface."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
from redisvl.utils.vectorize.base import BaseVectorizer

__all__ = ["OnnxVectorizer", "simple_char_code_tokenize"]


def simple_char_code_tokenize(text: str) -> list[int]:
    """A dependency-light placeholder tokenizer: char code mod a fixed vocab size.

    Not semantically meaningful — swap in a real HF tokenizer once a real
    embedding model artifact is deployed (see scripts/download_model.py).
    Good enough to make the cache pipeline exercise-able end to end today.
    """
    return [ord(c) % 1000 for c in text] or [0]


class OnnxVectorizer(BaseVectorizer):
    """Wraps an injected ONNX InferenceSession + tokenize function.

    Never constructs its own session — that's built once in the app's
    lifespan and shared across requests, same as the httpx/Redis clients.
    """

    model: str = "local-onnx"
    session: Any
    tokenize: Callable[[str], list[int]]

    def _embed(self, content: Any = "", **kwargs: Any) -> list[float]:
        """Run the sync ONNX inference call. Only ever call this from a worker thread."""
        token_ids = self.tokenize(content)
        input_array = np.array(token_ids, dtype=np.int64)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_array})
        result: list[float] = outputs[0].tolist()
        return result

    async def _aembed(self, content: Any = "", **kwargs: Any) -> list[float]:
        """Hand the sync ONNX call to a worker thread so it never blocks the event loop."""
        return await asyncio.to_thread(self._embed, content)
