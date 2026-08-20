"""Router-level test: pruning trims the payload sent upstream when over budget."""

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from fastapi_ctx_gateway.app import create_app
from fastapi_ctx_gateway.config import Settings, TokenBudgetConfig

pytestmark = pytest.mark.integration

TINY_BUDGET_MODEL = "gemini-3.7-flash"


def _settings(monkeypatch, token_budgets: TokenBudgetConfig) -> Settings:
    monkeypatch.setenv("GATEWAY_GEMINI_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_TENANT_API_KEYS", '{"gw-secret": "tenant-a"}')
    return Settings(token_budgets=token_budgets)


def _post(client: TestClient, turns: list[dict]) -> httpx.Response:
    return client.post(
        f"/v1/gemini/{TINY_BUDGET_MODEL}:streamGenerateContent",
        headers={"x-gateway-api-key": "gw-secret"},
        json={"turns": turns},
    )


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def test_over_budget_conversation_is_pruned_before_proxying(monkeypatch) -> None:
    tiny_budgets = TokenBudgetConfig(budgets={TINY_BUDGET_MODEL: 15}, default=15)
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post(f"/v1beta/models/{TINY_BUDGET_MODEL}:streamGenerateContent").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n")
        )

        app = create_app(_settings(monkeypatch, tiny_budgets))
        with TestClient(app) as client:
            response = _post(
                client,
                [
                    {"role": "user", "parts": [_text_part("oldest " * 20)]},
                    {"role": "assistant", "parts": [_text_part("middle " * 20)]},
                    {"role": "user", "parts": [_text_part("newest")]},
                ],
            )

    assert response.status_code == 200
    sent_body = json.loads(route.calls[0].request.content)
    sent_texts = [part["text"] for turn in sent_body["contents"] for part in turn["parts"]]
    assert sent_texts[-1] == "newest"
    assert "oldest " * 20 not in sent_texts


def test_under_budget_conversation_reaches_gemini_unmodified(monkeypatch) -> None:
    roomy_budgets = TokenBudgetConfig(budgets={TINY_BUDGET_MODEL: 100_000}, default=100_000)
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock:
        route = mock.post(f"/v1beta/models/{TINY_BUDGET_MODEL}:streamGenerateContent").mock(
            return_value=httpx.Response(200, content=b"data: {}\n\n")
        )

        app = create_app(_settings(monkeypatch, roomy_budgets))
        with TestClient(app) as client:
            response = _post(client, [{"role": "user", "parts": [_text_part("hi")]}])

    assert response.status_code == 200
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
