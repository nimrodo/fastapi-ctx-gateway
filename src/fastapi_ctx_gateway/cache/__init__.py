"""The semantic cache layer: RedisVL-backed lookup/store, gated by eligibility."""

from fastapi_ctx_gateway.cache.semantic_cache import CacheHit, SemanticCache, is_cache_eligible
from fastapi_ctx_gateway.cache.vectorizer import OnnxVectorizer

__all__ = ["CacheHit", "OnnxVectorizer", "SemanticCache", "is_cache_eligible"]
