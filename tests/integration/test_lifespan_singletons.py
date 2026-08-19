"""Multiple requests must never re-construct the lifespan-owned singletons."""

from pathlib import Path

import httpx
import onnxruntime as ort
import redis.asyncio as redis_asyncio
import respx
from fastapi.testclient import TestClient

import fastapi_ctx_gateway.app as app_module
from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings
from fastapi_ctx_gateway.proxy import client as client_module

FIXTURE_MODEL_PATH = str(
    Path(__file__).parent.parent / "fixtures" / "tiny_onnx_model" / "model.onnx"
)


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    monkeypatch.setenv("GATEWAY_EMBEDDING_MODEL_PATH", FIXTURE_MODEL_PATH)
    return Settings()


def test_singletons_are_built_exactly_once_across_multiple_requests(monkeypatch) -> None:
    calls = {"http_client": 0, "redis": 0, "onnx_session": 0, "gemini_client": 0}

    real_async_client_init = httpx.AsyncClient.__init__

    def counting_async_client_init(self, *args, **kwargs):
        calls["http_client"] += 1
        real_async_client_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", counting_async_client_init)

    real_from_url = redis_asyncio.from_url

    def counting_from_url(*args, **kwargs):
        calls["redis"] += 1
        return real_from_url(*args, **kwargs)

    monkeypatch.setattr(app_module.redis_asyncio, "from_url", counting_from_url)

    real_inference_session_init = ort.InferenceSession.__init__

    def counting_session_init(self, *args, **kwargs):
        calls["onnx_session"] += 1
        real_inference_session_init(self, *args, **kwargs)

    monkeypatch.setattr(ort.InferenceSession, "__init__", counting_session_init)

    real_gemini_client_init = client_module.GeminiClient.__init__

    def counting_gemini_client_init(self, *args, **kwargs):
        calls["gemini_client"] += 1
        real_gemini_client_init(self, *args, **kwargs)

    monkeypatch.setattr(app_module.GeminiClient, "__init__", counting_gemini_client_init)

    sse_body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},'
        b'"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":3}}\n\n'
    )
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        mock.post("/v1beta/models/gemini-2.5-flash:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            for i in range(3):
                response = client.post(
                    "/v1/gemini-2.5-flash:streamGenerateContent",
                    headers={"x-gateway-api-key": "gw-secret"},
                    json={"contents": [{"role": "user", "parts": [{"text": f"hi {i}"}]}]},
                )
                assert response.status_code == 200

    assert calls["http_client"] == 1
    assert calls["redis"] == 1
    assert calls["onnx_session"] == 1
    assert calls["gemini_client"] == 1
