"""Example: mounting fastapi-ctx-gateway as a sub-app inside a host FastAPI service.

Run with:

    uv run uvicorn examples.library_mount.app:app --app-dir . --reload

See README.md in this directory for setup and curl examples.
"""

from fastapi import FastAPI

from fastapi_ctx_gateway import Settings, create_app

app = FastAPI(title="host-app")


@app.get("/")
async def root() -> dict[str, str]:
    """The host app's own route — independent of anything the gateway defines."""
    return {"message": "host app; gateway mounted at /gateway"}


app.mount("/gateway", create_app(Settings()))
