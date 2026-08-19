"""Example: mounting fastapi-ctx-gateway as a sub-app inside a host FastAPI service.

Run with:

    uv run uvicorn examples.library_mount.app:app --app-dir . --reload

See README.md in this directory for setup and curl examples.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI

from fastapi_ctx_gateway import Settings, create_app

gateway_app = create_app(Settings())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Propagate startup/shutdown into the mounted gateway sub-app.

    Starlette does not do this automatically — without it, gateway_app.state
    (providers, rate_limiter, ...) is never populated and every request
    past /healthz 500s. Entering its lifespan here from the host's own
    lifespan is the standard workaround.
    """
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(gateway_app.router.lifespan_context(gateway_app))
        yield


app = FastAPI(title="host-app", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    """The host app's own route — independent of anything the gateway defines."""
    return {"message": "host app; gateway mounted at /gateway"}


app.mount("/gateway", gateway_app)
