"""The FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.proxy.client import GeminiClient
from fastapi_ctx_gateway.routers.generate import router as generate_router

__all__ = ["create_app"]


def create_app(settings: Settings) -> FastAPI:
    """Build a configured FastAPI application instance.

    No module-level app instance exists anywhere in this package — every
    caller (the CLI, tests, a consumer embedding this as a sub-app) builds
    its own via this factory, so independently-configured instances never
    share mutable state.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Built once per app instance and shared across all requests via
        # app.state — never per-request (connection setup cost alone would
        # blow the latency budget if paid on every call).
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            app.state.http_client = http_client
            app.state.gemini_client = GeminiClient(
                http_client=http_client,
                api_key=settings.gemini_upstream_key.get_secret_value(),
                base_url=settings.gemini_base_url,
            )
            yield

    app = FastAPI(title="fastapi-ctx-gateway", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(generate_router)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
