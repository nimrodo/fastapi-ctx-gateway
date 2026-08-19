"""Context-aware agentic API gateway for Gemini.

Public API surface: build a configured app via ``create_app(settings)``.
The CLI entrypoint (``main``) is intentionally not re-exported here — see
``fastapi_ctx_gateway.cli``.
"""

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

__all__ = ["create_app", "Settings"]
