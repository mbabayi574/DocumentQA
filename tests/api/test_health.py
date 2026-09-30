"""P0: app factory + liveness endpoint."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from qasystem.api.app import create_app
from qasystem.config import Settings, load_settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    settings = load_settings(env_file=None, embedding_provider="fake", app_env="test")
    assert isinstance(settings, Settings)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_is_always_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_needs_no_configuration(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_settings_are_available_on_app_state(client: TestClient) -> None:
    assert client.app.state.settings.embedding_provider == "fake"  # type: ignore[attr-defined]


def test_unknown_route_is_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]
