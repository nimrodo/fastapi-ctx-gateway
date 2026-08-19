"""The FastAPI application factory."""

from fastapi import FastAPI

from fastapi_ctx_gateway.config import Settings

__all__ = ["create_app"]


def create_app(settings: Settings) -> FastAPI:
    """Build a configured FastAPI application instance.

    No module-level app instance exists anywhere in this package — every
    caller (the CLI, tests, a consumer embedding this as a sub-app) builds
    its own via this factory, so independently-configured instances never
    share mutable state.
    """
    app = FastAPI(title="fastapi-ctx-gateway")
    app.state.settings = settings

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
