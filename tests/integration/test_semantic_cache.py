"""Integration tests for SemanticCache against real Redis Stack (RediSearch/VSS)."""

from pathlib import Path

import onnxruntime as ort
import pytest
from redisvl.extensions.cache.llm import SemanticCache as RedisVLSemanticCache

from fastapi_ctx_gateway.cache.semantic_cache import SemanticCache
from fastapi_ctx_gateway.cache.vectorizer import OnnxVectorizer, simple_char_code_tokenize
from fastapi_ctx_gateway.schemas.gemini import Content, Part

FIXTURE_MODEL_PATH = Path(__file__).parent.parent / "fixtures" / "tiny_onnx_model" / "model.onnx"

pytestmark = pytest.mark.integration


def _contents(text: str) -> list[Content]:
    return [Content(role="user", parts=[Part(text=text)])]


@pytest.fixture
def vectorizer() -> OnnxVectorizer:
    session = ort.InferenceSession(str(FIXTURE_MODEL_PATH))
    return OnnxVectorizer(session=session, tokenize=simple_char_code_tokenize, dims=384)


@pytest.fixture
async def cache(vectorizer: OnnxVectorizer) -> SemanticCache:
    redis_cache = RedisVLSemanticCache(
        name="test_semantic_cache",
        distance_threshold=0.05,
        ttl=60,
        vectorizer=vectorizer,
        filterable_fields=[
            {"name": "tenant_id", "type": "tag"},
            {"name": "model", "type": "tag"},
        ],
        redis_url="redis://localhost:6379",
        overwrite=True,
    )
    wrapper = SemanticCache(
        redis_cache=redis_cache,
        vectorizer=vectorizer,
        temperature_threshold=0.3,
        lookup_timeout_s=1.0,
    )
    yield wrapper
    await redis_cache.adelete()


async def test_store_then_lookup_is_a_hit(cache: SemanticCache) -> None:
    contents = _contents("what is the capital of france")
    await cache.store(contents, "tenant-a", "model-x", "Paris", {"totalTokenCount": 5})
    hit = await cache.lookup(contents, "tenant-a", "model-x")
    assert hit is not None
    assert hit.response_text == "Paris"
    assert hit.usage == {"totalTokenCount": 5}


async def test_dissimilar_prompt_is_a_miss(cache: SemanticCache) -> None:
    await cache.store(
        _contents("what is the capital of france"), "tenant-a", "model-x", "Paris", None
    )
    hit = await cache.lookup(_contents("write a poem about the ocean"), "tenant-a", "model-x")
    assert hit is None


async def test_different_tenant_is_a_miss(cache: SemanticCache) -> None:
    contents = _contents("what is the capital of france")
    await cache.store(contents, "tenant-a", "model-x", "Paris", None)
    hit = await cache.lookup(contents, "tenant-b", "model-x")
    assert hit is None


async def test_different_model_is_a_miss(cache: SemanticCache) -> None:
    contents = _contents("what is the capital of france")
    await cache.store(contents, "tenant-a", "model-x", "Paris", None)
    hit = await cache.lookup(contents, "tenant-a", "model-y")
    assert hit is None
