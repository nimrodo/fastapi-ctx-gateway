"""Integration test: the agent-provider example's app works end to end.

Complements tests/unit/test_examples_agent_provider.py, which only checks
that the app boots (no Redis) — this exercises both registered routes for
real, same as test_stream_endpoint_agent.py does for the provider itself.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

_APP_PATH = (Path(__file__).parents[2] / "examples" / "agent_provider" / "app.py").resolve()


def _post(client: TestClient, path: str, text: str = "hi") -> Any:
    return client.post(
        path,
        headers={"x-gateway-api-key": "test-key"},
        json={"turns": [{"role": "user", "parts": [{"type": "text", "text": text}]}]},
    )


def test_echo_agent_streams_input_back(load_example_app) -> None:
    app = load_example_app(_APP_PATH, "agent_provider_example_app")
    with TestClient(app) as client:
        response = _post(client, "/v1/echo/default:streamGenerateContent", text="hello there")

    assert response.status_code == 200
    assert b'"text":"hello "' in response.content
    assert b'"text":"there "' in response.content
    assert b'"finish_reason":"stop"' in response.content


def test_research_agent_streams_intermediate_step_before_text(load_example_app) -> None:
    app = load_example_app(_APP_PATH, "agent_provider_example_app")
    with TestClient(app) as client:
        response = _post(client, "/v1/research/default:streamGenerateContent")

    assert response.status_code == 200
    assert b'"intermediate":{"label":"tool_call"' in response.content
    assert b"Found" in response.content
    assert b"results." in response.content
    intermediate_index = response.content.index(b'"intermediate"')
    text_index = response.content.index(b'"text":"Found')
    assert intermediate_index < text_index
