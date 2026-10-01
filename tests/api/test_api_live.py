"""P8 live: the HTTP surface against the real provider.

Skipped unless ``RUN_LIVE=1``. Everything below the HTTP layer is already covered live by
``test_retrieval_live.py``; what this file adds is the **wiring** -- that the service graph
probes the real model, that a real upload produces a real answer through a real HTTP
response, and that readiness reports the real model rather than a cached guess.

It uses ``TestClient`` rather than a live socket on purpose: the thing under test is the app's
own composition, not uvicorn's transport.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from qasystem.api.app import create_app
from qasystem.config import load_settings
from qasystem.errors import ConfigError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
PERSIAN_PDF = FIXTURES / "fa" / "ai-engineer.pdf"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real provider"
)


@pytest.fixture
def live_client(tmp_path: Path) -> Any:
    """The app builds its own graph, in its own event loop, exactly as uvicorn does.

    Injecting a graph built elsewhere cannot work here: ``httpx.AsyncClient`` binds to the
    loop that created it, and ``TestClient`` runs a separate one. That constraint is worth
    testing *around* rather than against -- so this file goes through ``create_app()`` with no
    injected services and only redirects the data paths, which is the production path plus a
    tmpdir.
    """
    try:
        base = load_settings()
    except ConfigError as exc:
        pytest.skip(f"live run needs a configured provider: {exc.code}")
    if base.is_fake_provider:
        pytest.skip("EMBEDDING_PROVIDER=fake; nothing live to call")

    settings = base.model_copy(
        update={
            "data_dir": tmp_path,
            "sqlite_path": tmp_path / "qasystem.db",
            "chroma_path": tmp_path / "chroma",
        }
    )
    with TestClient(create_app(settings=settings)) as client:
        yield client


def upload(path: Path) -> dict[str, Any]:
    content_type = "application/pdf" if path.suffix == ".pdf" else "text/markdown"
    return {"file": (path.name, path.read_bytes(), content_type)}


def test_the_graph_probes_the_real_model_and_ready_reports_it(live_client: Any) -> None:
    """L9: readiness reports what the provider actually serves, without re-probing."""
    body = live_client.get("/ready").json()

    assert body["status"] == "ready"
    assert body["embedder"] == "remote"
    assert body["model_id"] and body["dimension"] > 0
    # The shipped thresholds belong to the real model, so they load rather than fall back.
    assert body["thresholds_calibrated"] is False


def test_a_real_upload_and_query_round_trip(live_client: Any) -> None:
    created = live_client.post("/documents", files=upload(PERSIAN_PDF))
    assert created.status_code == 201, created.text
    assert created.json()["chunks_added"] > 0
    assert created.json()["embed_requests"] >= 1

    answer = live_client.post("/query", json={"question": "BGE-M3"}).json()

    assert answer["status"] == "answered", answer
    assert answer["citations"][0]["doc_id"] == "ai-engineer"
    assert answer["citations"][0]["document"] == "ai-engineer.pdf"
    # A real provider call really happened, and the answer really came from the PDF.
    assert answer["citations"][0]["page"] is not None
    assert answer["citations"][0]["excerpt"]


def test_the_ingest_log_counts_real_network_requests(live_client: Any) -> None:
    """D46, over HTTP: the number in the log is the number the provider was asked for."""
    created = live_client.post("/documents", files=upload(PERSIAN_PDF)).json()

    # 7 chunks is one batch, so one request; if this ever exceeds 1 the batching changed and
    # the figure means something different.
    assert created["embed_requests"] == 1
    assert created["chunks_added"] > 1, "more than one chunk must share that one request"


def test_put_then_delete_over_http_keeps_stale_content_unreachable(live_client: Any) -> None:
    """I1-I3 through the real stack: an edit then a delete, with real vectors."""
    live_client.post("/documents", files=upload(PERSIAN_PDF))
    assert live_client.post("/query", json={"question": "BGE-M3"}).json()["status"] == "answered"

    assert live_client.delete("/documents/ai-engineer").status_code == 204
    assert live_client.delete("/documents/ai-engineer").status_code == 404

    answer = live_client.post("/query", json={"question": "BGE-M3"}).json()
    assert answer["status"] == "insufficient_information"
    assert answer["citations"] == []
    assert live_client.get("/documents").json()["total"] == 0


def test_an_unanswerable_question_is_refused_over_http(live_client: Any) -> None:
    live_client.post("/documents", files=upload(PERSIAN_PDF))

    body = live_client.post(
        "/query", json={"question": "what is the boiling point of mercury at 2000 metres"}
    ).json()

    assert body["status"] == "insufficient_information"
    assert not body["citations"]
    assert body["evidence_score"] == 0.0


def test_no_secret_reaches_a_response_over_the_real_stack(live_client: Any) -> None:
    """I8 with a real provider in the graph, where provider errors are actually possible."""
    token = load_settings().embedding_api_key
    if token is None:
        pytest.skip("no token configured, so there is nothing to assert is absent")
    secret = token.get_secret_value()

    live_client.post("/documents", files=upload(PERSIAN_PDF))
    live_client.post("/documents", files={"file": ("x.exe", b"nope", "application/octet-stream")})

    for response in (
        live_client.get("/ready"),
        live_client.get("/documents"),
        live_client.post("/query", json={"question": "BGE-M3", "debug": True}),
        live_client.post("/query", json={"question": "mercury boiling point"}),
        live_client.get("/documents/missing"),
    ):
        assert secret not in response.text, response.request.url


def test_a_large_real_document_ingests_within_one_request_batch(
    live_client: Any, tmp_path: Path
) -> None:
    """The 393k-char Persian book: 1025 chunks, so the batching bound has to hold for real."""
    book = FIXTURES / "fa" / "justforfun_book_a4.pdf"
    if not book.exists():
        pytest.skip("the Persian book fixture is not committed")

    created = live_client.post("/documents", files=upload(book))

    assert created.status_code == 201, created.text[:400]
    body = created.json()
    assert body["chunks_added"] > 1000
    # 1025 chunks at MAX_ITEMS_PER_BATCH=32 is 33 requests, not one and not 1025.
    assert 32 <= body["embed_requests"] <= 34, body["embed_requests"]
