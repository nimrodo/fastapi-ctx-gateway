"""The pluggable provider contract: neutral request in, neutral SSE bytes out."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from fastapi_ctx_gateway.schemas.neutral import NeutralGenerateRequest

__all__ = ["Provider"]


class Provider(ABC):
    """Translates the gateway's neutral contract to/from one upstream LLM API.

    `name` is the path segment clients use to select this provider
    (`/v1/{name}/{model}:streamGenerateContent`).
    """

    name: str

    @abstractmethod
    def stream(self, model: str, request: NeutralGenerateRequest) -> AsyncIterator[bytes]:
        r"""Call upstream and yield neutral-schema SSE bytes.

        One upstream event translates to exactly one yielded neutral SSE
        event (`data: {...}\n\n`) — chunk boundaries are never re-buffered
        or re-split. Never raises for upstream or transport failures; those
        are translated in-band to a terminal `NeutralErrorEvent` chunk
        instead, so the caller always gets a well-formed stream to forward.
        """
        raise NotImplementedError
