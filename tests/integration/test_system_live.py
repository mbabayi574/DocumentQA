"""Whole-system live verification against the real provider.

Skipped unless ``RUN_LIVE=1``. Everything here exists because the previous live files left a
gap. P8 made the HTTP surface run against real vectors, but these were still unverified:

* **The CLI.** ``check-storage``, ``reconcile`` and ``rebuild`` had never touched the real
  provider -- and ``rebuild`` is the one command whose whole promise is that it needs no API.
* **I10 measured.** ``rebuild`` was only ever asserted against the fake embedder, where
  "zero requests" is vacuous. Here it runs on vectors BGE-M3 actually produced, and the
  request counter is read off the real client.
* **The full fixture corpus.** 1225 chunks, 84% of them one Persian book. Every previous live
  test used one 7-chunk PDF, which does not exercise batching, throughput, or the skew.
  **Removed in P11** as the suite's one intermittent failure (D80/D82): it asserted exact top-1
  ranks against Chroma's approximate HNSW index, which is the D56 defect class. The batching
  and at-scale figures it produced are recorded in D70/D71 and stand as measurements; what is
  gone is the *regression* witness, which is stated plainly in D82 rather than papered over.
* **P6's invariants over real vectors.** Local-edit isolation and the reorder-only zero-cost
  property were only ever checked against hashed tokens.

Each test builds its own graph on ``tmp_path``, because Chroma caches a collection per path for
the life of the process (D28) and sharing one would let a test inherit another's index.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from qasystem.api.app import create_app
from qasystem.api.deps import build_services
from qasystem.config import load_settings
from qasystem.errors import ConfigError
from qasystem.ingestion.reconcile import rebuild
from qasystem.storage.chroma_store import ChromaStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"
PERSIAN_PDF = FIXTURES / "fa" / "ai-engineer.pdf"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real provider"
)


def _settings(tmp_path: Path) -> Any:
    try:
        base = load_settings()
    except ConfigError as exc:
        pytest.skip(f"live run needs a configured provider: {exc.code}")
    if base.is_fake_provider:
        pytest.skip("EMBEDDING_PROVIDER=fake; nothing live to call")
    return base.model_copy(
        update={
            "data_dir": tmp_path,
            "sqlite_path": tmp_path / "qasystem.db",
            "chroma_path": tmp_path / "chroma",
        }
    )


@pytest.fixture
async def graph(tmp_path: Path) -> Any:
    services = await build_services(_settings(tmp_path))
    try:
        yield services
    finally:
        await services.aclose()


async def ingest(services: Any, path: Path, doc_id: str | None = None) -> Any:
    return await services.ingestion.ingest(path.read_bytes(), path.name, doc_id=doc_id)


# ---------------------------------------------------------------- I10, measured


async def test_rebuild_restores_real_vectors_with_zero_api_calls(graph: Any) -> None:
    """I10 on vectors BGE-M3 actually produced.

    The fake embedder made "zero requests" vacuous: it makes a request for anything. Here the
    counters belong to the real client, so this is the assertion the plan actually asks for --
    deleting the index and restoring it must cost nothing at the provider.
    """
    await ingest(graph, PERSIAN_PDF)
    await ingest(graph, FIXTURES / "en" / "storyen.md")
    expected = sorted(graph.vectors.list_ids())
    assert expected

    client = graph._client
    assert client is not None, "the fake provider has no HTTP client to measure"

    # An empty index in a fresh directory: the strictest form of "the index is disposable".
    empty = ChromaStore(
        graph.settings.chroma_path.parent / "chroma-empty",
        model_id=graph.model_id,
        dimension=graph.dimension,
    )
    empty.ensure_collection()
    before = client.requests
    restored = rebuild(graph.store, empty, graph.model_id)
    after = client.requests

    assert restored == len(expected)
    assert sorted(empty.list_ids()) == expected
    assert after == before, f"rebuild spent {after - before} API request(s); it must spend none"
    empty.close()


async def test_rebuild_into_a_deleted_chroma_directory_works(graph: Any, tmp_path: Path) -> None:
    """L8's backup story: stop the app, copy data/, and dense search comes back."""
    await ingest(graph, PERSIAN_PDF)
    found_before = graph.vectors.count()
    assert found_before > 0

    restored = rebuild(graph.store, graph.vectors, graph.model_id)

    assert restored > 0
    assert graph.vectors.count() == found_before


# ---------------------------------------------------------------- P6 over real vectors


async def test_a_local_edit_embeds_only_the_changed_chunk(graph: Any) -> None:
    """I5 with real vectors: the unchanged chunks must cost no provider request."""
    original = (
        "# Handbook\n\n## Install\n\nRun the installer on Linux first.\n\n"
        "## Usage\n\nThe service returns ERR-404 for an invalid token.\n"
    )
    edited = original.replace("ERR-404", "ERR-503")

    first = await graph.ingestion.ingest(original.encode(), "handbook.md")
    assert first.chunks_added == 2

    client = graph._client
    before = client.requests
    second = await graph.ingestion.ingest(edited.encode(), "handbook.md")
    spent = client.requests - before

    assert second.status == "updated"
    assert second.chunks_added == 1, "only the edited paragraph is new"
    assert second.chunks_reused == 1
    assert spent == 1, f"a one-paragraph edit spent {spent} requests, expected exactly 1"


async def test_a_reorder_only_edit_costs_zero_requests(graph: Any) -> None:
    """Plan P6 rule 3: reordering sections must not cost a provider request."""
    original = "# G\n\n## A\n\nAlpha text here.\n\n## B\n\nBeta text here.\n"
    reordered = "# G\n\n## B\n\nBeta text here.\n\n## A\n\nAlpha text here.\n"

    await graph.ingestion.ingest(original.encode(), "guide.md")
    client = graph._client
    before = client.requests

    result = await graph.ingestion.ingest(reordered.encode(), "guide.md")

    assert result.status == "updated"
    assert client.requests == before, (
        f"a reorder spent {client.requests - before} provider request(s); it must spend none"
    )
    assert result.chunks_reused == 2


async def test_identical_bytes_cost_zero_requests(graph: Any) -> None:
    """I4 measured on the real client rather than asserted on a diff."""
    await ingest(graph, PERSIAN_PDF)
    client = graph._client
    before = client.requests

    again = await ingest(graph, PERSIAN_PDF)

    assert again.status == "unchanged"
    assert client.requests == before
    assert again.embed_requests == 0


# ---------------------------------------------------------------- the full corpus


# ---------------------------------------------------------------- the CLI


def _cli_env(tmp_path: Path) -> dict[str, str]:
    import os as _os

    environment = dict(_os.environ)
    environment.update(
        {
            "DATA_DIR": str(tmp_path),
            "SQLITE_PATH": str(tmp_path / "qasystem.db"),
            "CHROMA_PATH": str(tmp_path / "chroma"),
        }
    )
    return environment


def _cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "qasystem.cli", *args],
        capture_output=True,
        text=True,
        timeout=600,
        env=_cli_env(tmp_path),
    )


def test_the_cli_check_storage_passes_against_the_real_provider(tmp_path: Path) -> None:
    result = _cli(tmp_path, "check-storage")
    body = json.loads(result.stdout)

    assert result.returncode == 0, result.stdout + result.stderr[-400:]
    assert body["round_trip_ok"] is True
    assert body["model_id"] == "Bge-m3"
    assert body["dimension"] == 1024


def test_the_cli_reconcile_and_rebuild_run_against_the_real_provider(tmp_path: Path) -> None:
    ingest_script = """
import sys
from pathlib import Path
from qasystem.api.deps import build_services
from qasystem.config import load_settings
import anyio

data = sys.argv[1]
document = Path(sys.argv[2])
settings = load_settings()
settings = settings.model_copy(update={
    "data_dir": Path(data),
    "sqlite_path": Path(data) / "qasystem.db",
    "chroma_path": Path(data) / "chroma",
})

async def go():
    services = await build_services(settings)
    await services.ingestion.ingest(document.read_bytes(), document.name)
    await services.aclose()

anyio.run(go)
"""
    subprocess.run(
        [sys.executable, "-c", ingest_script, str(tmp_path), str(PERSIAN_PDF)],
        capture_output=True,
        text=True,
        timeout=600,
        env=_cli_env(tmp_path),
        check=True,
    )

    reconciled = _cli(tmp_path, "reconcile")
    assert reconciled.returncode == 0, reconciled.stdout + reconciled.stderr[-400:]
    assert json.loads(reconciled.stdout)["in_sync"] is True

    rebuilt = _cli(tmp_path, "rebuild")
    assert rebuilt.returncode == 0, rebuilt.stdout + rebuilt.stderr[-400:]
    body = json.loads(rebuilt.stdout)
    assert body["model_id"] == "Bge-m3"
    assert body["restored"] > 0


# ---------------------------------------------------------------- the app end to end


def test_the_app_serves_the_corpus_end_to_end_over_real_http(tmp_path: Path) -> None:
    """`scripts/smoke_test.sh` has only ever run against the fake provider. Not any more."""
    from fastapi.testclient import TestClient

    settings = _settings(tmp_path)
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/ready").json()["embedder"] == "remote"

        created = client.post(
            "/documents",
            files={"file": (PERSIAN_PDF.name, PERSIAN_PDF.read_bytes(), "application/pdf")},
        )
        assert created.status_code == 201, created.text[:400]
        assert created.json()["chunks_added"] > 0

        answer = client.post("/query", json={"question": "پردازش اسناد"}).json()
        assert answer["status"] == "answered", answer
        assert answer["citations"][0]["page"] is not None


# ---------------------------------------------------------------- measured, not hoped for


async def test_query_latency_is_the_embedding_api_not_the_retrieval(graph: Any) -> None:
    """§9.2 wants P50/P95 query latency. Measured here so P9 starts from a number.

    On the 7-chunk Persian PDF with real BGE-M3: one embedding call measures 300-450ms, while
    retrieval, fusion, the gate and answer assembly together measure roughly 20-40ms. So query
    latency is almost entirely one network round trip, and optimising retrieval buys almost
    nothing -- batching and serving the embedder is where the time is.

    Measured through the public surface: the query is embedded once to warm the cache, and then
    ``retrieve()`` does its own embedding as a cache hit, leaving the local work measurable
    without touching a private attribute.
    """
    await ingest(graph, PERSIAN_PDF)
    service = graph.retrieval
    local_ms: list[float] = []

    for question in ("BGE-M3", "پردازش اسناد", "مدیریت تغییرات", "مستندات OpenAPI"):
        await graph.embedder.embed([question])  # the network call, deliberately not measured
        started = time.perf_counter()
        candidates = await service.retrieve(question, limit=30)
        local_ms.append((time.perf_counter() - started) * 1000)
        assert candidates

    worst = max(local_ms)
    assert worst < 150, f"local retrieval work took {worst:.0f}ms against a ~400ms network call"


async def test_the_cross_lingual_miss_is_the_lexical_arm_not_the_dense_bar(graph: Any) -> None:
    """Pins the D47 trade-off with numbers, so changing it has to be a decision.

    An English question whose answer is in the Persian PDF is retrieved correctly *by the dense
    arm* and then out-ranked by English chunks, purely because those share incidental words and
    so collect lexical ranks the Persian chunk can never collect. No dense-similarity bar
    separates them: measured, the wrong chunk sits at 0.547 and the right one at 0.539.

    So this is a fusion-weight question (plan.md section 9.4, experiment 1) and not a threshold
    bug. This test records today's behaviour; P9 changes it deliberately.
    """
    await ingest(graph, PERSIAN_PDF)
    await ingest(graph, FIXTURES / "en" / "clean-code-excerpt.pdf")
    service = graph.retrieval

    candidates = await service.retrieve("what language and tools are free to use", limit=30)
    persian = [c for c in candidates if c.source_name == PERSIAN_PDF.name]

    # The correct chunk is retrieved, and it is near the top of the *dense* arm.
    assert persian, "the Persian source was not retrieved at all"
    best_persian = max(persian, key=lambda c: c.similarity or 0.0)
    assert (best_persian.similarity or 0.0) > 0.5, best_persian.similarity
    assert best_persian.lexical_score is None, (
        "a cross-lingual chunk can never collect a lexical rank; if it can, this test's "
        "explanation is wrong and the trade-off needs re-measuring"
    )
    # And it still loses the fused ranking to English chunks that share incidental words.
    assert candidates[0].source_name != PERSIAN_PDF.name, (
        "the lexical arm no longer misleads this query -- re-measure before trusting the "
        "conclusion in D54"
    )
