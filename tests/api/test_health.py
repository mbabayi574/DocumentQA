"""P0: app factory + liveness endpoint."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qasystem.api.app import create_app
from qasystem.config import Settings, load_settings


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """The fake embedder, and a thresholds file that belongs to it.

    P9 made the shipped ``config/thresholds.json`` a *calibrated* file for ``Bge-m3``, and a
    calibrated file for another model is a hard ``ConfigError`` (D51) rather than a
    fallback. That is the rule working, not a broken fixture: an offline test on the fake
    embedder must supply its own numbers, exactly as a real deployment with a different
    model would have to.
    """
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "version": 1,
                "model_id": "fake-embedder",
                "calibrated": False,
                "min_dense": 0.05,
                "min_dense_alone": 0.99,
                "min_coverage": 0.70,
                "min_coverage_high": 0.90,
                "min_sentence_overlap": 0.15,
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(
        env_file=None,
        embedding_provider="fake",
        app_env="test",
        data_dir=tmp_path,
        sqlite_path=tmp_path / "qasystem.db",
        chroma_path=tmp_path / "chroma",
        thresholds_path=thresholds,
    )
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
