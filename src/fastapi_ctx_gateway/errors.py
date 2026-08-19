"""Shared exception -> HTTP response mappings, registered on the app in app.py."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastapi_ctx_gateway.circuit_breaker import CircuitOpenError
from fastapi_ctx_gateway.ratelimit import RateLimitExceeded
from fastapi_ctx_gateway.schemas.neutral import NeutralError, NeutralErrorEvent

__all__ = ["ProviderNotFoundError", "register_exception_handlers"]


class ProviderNotFoundError(Exception):
    """Raised when a request names a provider path segment the gateway doesn't know."""

    def __init__(self, provider_name: str) -> None:
        """Carry the unknown name so the handler can report it back to the client."""
        self.provider_name = provider_name
        super().__init__(f"unknown provider: {provider_name}")


def _error_body(message: str, error_type: str) -> dict:
    event = NeutralErrorEvent(error=NeutralError(message=message, type=error_type))
    return event.model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    """Wire domain exceptions to their HTTP responses."""

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        del request
        retry_after = max(1, round(exc.decision.retry_after_s))
        return JSONResponse(
            status_code=429,
            content=_error_body(str(exc), "rate_limited"),
            headers={"Retry-After": str(retry_after)},
        )

    @app.exception_handler(CircuitOpenError)
    async def _circuit_open(request: Request, exc: CircuitOpenError) -> JSONResponse:
        del request, exc
        return JSONResponse(
            status_code=503,
            content=_error_body("upstream temporarily unavailable", "circuit_open"),
        )

    @app.exception_handler(ProviderNotFoundError)
    async def _provider_not_found(request: Request, exc: ProviderNotFoundError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=404,
            content=_error_body(str(exc), "provider_not_found"),
        )
