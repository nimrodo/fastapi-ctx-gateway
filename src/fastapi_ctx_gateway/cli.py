"""The `fastapi-ctx-gateway` CLI entrypoint.

Kept separate from the importable library surface (`__init__.py`) so that
importing the package never has the side effect of parsing the environment
or being runnable as a server.
"""

import uvicorn

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

__all__ = ["main"]


def main() -> None:
    """Build the app from environment-loaded settings and run the server."""
    settings = Settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, workers=settings.workers)
