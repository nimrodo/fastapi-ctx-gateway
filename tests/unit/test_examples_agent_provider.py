"""Tests for the agent-provider example under examples/agent_provider/.

Stays a unit test by checking only that the app boots and both agent
providers are registered — an actual `/streamGenerateContent` call needs
real Redis (rate limiting), so that end-to-end path is covered instead by
tests/integration/test_examples_agent_provider.py, matching how
test_stream_endpoint_agent.py covers the underlying provider.
"""

from pathlib import Path

from fastapi.testclient import TestClient

_APP_PATH = (Path(__file__).parents[2] / "examples" / "agent_provider" / "app.py").resolve()


def test_healthz_is_reachable(load_example_app) -> None:
    app = load_example_app(_APP_PATH, "agent_provider_example_app")
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200


def test_both_agent_providers_are_registered(load_example_app) -> None:
    app = load_example_app(_APP_PATH, "agent_provider_example_app")
    with TestClient(app):
        assert set(app.state.providers) == {"gemini", "echo", "research"}
