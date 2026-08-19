"""Router-level test: cache hit/miss lifecycle end to end (real Redis Stack)."""

import json
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from support.neutral_sse import gemini_sse_event

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings

pytestmark = pytest.mark.integration

MODEL = "gemini-3.7-flash"
FIXTURE_MODEL_PATH = str(
    Path(__file__).parent.parent / "fixtures" / "tiny_onnx_model" / "model.onnx"
)


def _settings(monkeypatch) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    monkeypatch.setenv("GATEWAY_EMBEDDING_MODEL_PATH", FIXTURE_MODEL_PATH)
    monkeypatch.setenv("GATEWAY_CACHE_DISTANCE_THRESHOLD", "0.05")
    return Settings()


def _post(client: TestClient, text: str, temperature: float = 0.0) -> httpx.Response:
    return client.post(
        f"/v1/gemini/{MODEL}:streamGenerateContent",
        headers={"x-gateway-api-key": "gw-secret"},
        json={
            "turns": [{"role": "user", "parts": [{"type": "text", "text": text}]}],
            "generation_config": {"temperature": temperature},
        },
    )


def test_second_identical_request_is_a_cache_hit_and_skips_gemini(monkeypatch) -> None:
    sse_body = gemini_sse_event(text="Paris", finish_reason="STOP", total_tokens=7)
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post(f"/v1beta/models/{MODEL}:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            first = _post(client, "what is the capital of france")
            second = _post(client, "what is the capital of france")

    assert first.status_code == 200
    assert first.headers.get("x-cache") == "MISS"
    assert second.status_code == 200
    assert second.headers.get("x-cache") == "HIT"
    assert route.call_count == 1  # Gemini only called once, for the miss

    hit_body = json.loads(second.content.removeprefix(b"data: ").rstrip(b"\n"))
    hit_text = hit_body["delta"]["parts"][0]["text"]
    assert hit_text == "Paris"


def test_stream_without_finish_reason_is_never_cached(monkeypatch) -> None:
    # No finishReason chunk at all -> tee_stream never sees a clean finish.
    truncated_sse_body = gemini_sse_event(text="partial")
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post(f"/v1beta/models/{MODEL}:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=truncated_sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            first = _post(client, "a prompt that never completes cleanly")
            second = _post(client, "a prompt that never completes cleanly")

    assert first.headers.get("x-cache") == "MISS"
    assert second.headers.get("x-cache") == "MISS"  # still a miss - nothing was ever stored
    assert route.call_count == 2  # Gemini called again, since nothing got cached


def test_high_temperature_request_bypasses_cache(monkeypatch) -> None:
    sse_body = gemini_sse_event(text="a random poem", finish_reason="STOP", total_tokens=9)
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post(f"/v1beta/models/{MODEL}:streamGenerateContent").mock(
            return_value=httpx.Response(
                200, content=sse_body, headers={"content-type": "text/event-stream"}
            )
        )

        app = create_app(_settings(monkeypatch))
        with TestClient(app) as client:
            first = _post(client, "write me something creative", temperature=0.9)
            second = _post(client, "write me something creative", temperature=0.9)

    assert first.headers.get("x-cache") == "MISS"
    assert second.headers.get("x-cache") == "MISS"
    assert route.call_count == 2
