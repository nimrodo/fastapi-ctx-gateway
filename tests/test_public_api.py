"""Import-surface tests for the package's public API."""

import fastapi_ctx_gateway


def test_create_app_is_exported() -> None:
    assert hasattr(fastapi_ctx_gateway, "create_app")


def test_settings_is_exported() -> None:
    assert hasattr(fastapi_ctx_gateway, "Settings")


def test_reusable_components_are_exported() -> None:
    for name in (
        "SemanticCache",
        "OnnxVectorizer",
        "RateLimiter",
        "TokenEstimator",
        "TokenBudgetPruner",
        "CircuitBreaker",
    ):
        assert hasattr(fastapi_ctx_gateway, name), f"{name} should be importable standalone"


def test_main_is_not_exported() -> None:
    """The CLI entrypoint stays out of the importable library surface."""
    assert not hasattr(fastapi_ctx_gateway, "main")
