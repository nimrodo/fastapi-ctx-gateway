"""Table tests for SemanticCache.is_eligible."""

import pytest

from fastapi_ctx_gateway.cache.semantic_cache import is_cache_eligible
from fastapi_ctx_gateway.schemas.gemini import GenerationConfig

THRESHOLD = 0.3


@pytest.mark.parametrize(
    ("tools", "temperature", "expected"),
    [
        (None, None, False),  # unset temperature is treated as "not explicitly low"
        (None, 0.0, True),
        (None, 0.3, True),
        (None, 0.31, False),
        (None, 1.0, False),
        ([{"functionDeclarations": []}], 0.0, False),  # tools present -> always bypass
        ([{"functionDeclarations": []}], None, False),
    ],
)
def test_eligibility_table(tools, temperature, expected) -> None:
    generation_config = (
        GenerationConfig(temperature=temperature) if temperature is not None else None
    )
    eligible = is_cache_eligible(
        tools=tools, generation_config=generation_config, temperature_threshold=THRESHOLD
    )
    assert eligible is expected
