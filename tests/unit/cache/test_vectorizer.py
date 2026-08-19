"""Tests for OnnxVectorizer against the tiny fixture ONNX model."""

import threading
from pathlib import Path

import onnxruntime as ort
import pytest

from fastapi_ctx_gateway.cache.vectorizer import OnnxVectorizer

FIXTURE_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "fixtures" / "tiny_onnx_model" / "model.onnx"
)


def _tiny_tokenize(text: str) -> list[int]:
    return [ord(c) % 1000 for c in text] or [0]


@pytest.fixture
def vectorizer() -> OnnxVectorizer:
    session = ort.InferenceSession(str(FIXTURE_MODEL_PATH))
    return OnnxVectorizer(session=session, tokenize=_tiny_tokenize, dims=384)


def test_embed_returns_384_dim_vector(vectorizer: OnnxVectorizer) -> None:
    vector = vectorizer.embed("hello world")
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


def test_embed_is_deterministic(vectorizer: OnnxVectorizer) -> None:
    assert vectorizer.embed("hello") == vectorizer.embed("hello")


def test_embed_differs_for_different_text(vectorizer: OnnxVectorizer) -> None:
    assert vectorizer.embed("hello") != vectorizer.embed("goodbye")


async def test_aembed_returns_same_result_as_embed(vectorizer: OnnxVectorizer) -> None:
    assert await vectorizer.aembed("hello world") == vectorizer.embed("hello world")


async def test_aembed_runs_off_the_event_loop_thread() -> None:
    """aembed must hand the sync ONNX call to a worker thread, never run it inline."""
    main_thread_id = threading.get_ident()
    seen_thread_id: int | None = None

    def spy_tokenize(text: str) -> list[int]:
        nonlocal seen_thread_id
        seen_thread_id = threading.get_ident()
        return _tiny_tokenize(text)

    session = ort.InferenceSession(str(FIXTURE_MODEL_PATH))
    vectorizer = OnnxVectorizer(session=session, tokenize=spy_tokenize, dims=384)

    await vectorizer.aembed("hello")

    assert seen_thread_id is not None
    assert seen_thread_id != main_thread_id
