"""Shared exception -> HTTP response mappings, registered on the app in app.py."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastapi_ctx_gateway.circuit_breaker import CircuitOpenError
from fastapi_ctx_gateway.ratelimit import RateLimitExceeded

__all__ = ["register_exception_handlers"]


def register_exception_handlers(app: FastAPI) -> None:
    """Wire domain exceptions to their HTTP responses."""

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        del request
        retry_after = max(1, round(exc.decision.retry_after_s))
        return JSONResponse(
            status_code=429,
            content={"error": {"message": str(exc), "status": "RESOURCE_EXHAUSTED"}},
            headers={"Retry-After": str(retry_after)},
        )

    @app.exception_handler(CircuitOpenError)
    async def _circuit_open(request: Request, exc: CircuitOpenError) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=503,
            content={
                "error": {"message": "upstream temporarily unavailable", "status": "UNAVAILABLE"}
            },
        )
