"""P5: reconcile planning and ``rebuild`` (I10), plus the single-owner lock (L2).

``rebuild`` is the proof that Chroma is disposable: it must restore dense search from
SQLite with zero embedding calls. The test asserts the call count, not just the result.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from qasystem.domain.models import Chunk
from qasystem.embeddings.caching import input_hash
from qasystem.errors import StorageLockedError
from qasystem.ingestion.reconcile import plan_reconcile, rebuild
from qasystem.storage.chroma_store import ChromaStore
from qasystem.storage.lock import DataLock
from qasystem.storage.sqlite_store import SqliteStore

# L2 needs real processes. The holder keeps a *named* reference to its DataLock: a
# temporary is garbage-collected mid-sleep and releases the lock, which made an earlier
# version of this test pass for the wrong reason.
HOLDER = (
    "import sys, time\n"
    "from qasystem.storage.lock import DataLock\n"
    "lock = DataLock(sys.argv[1])\n"
    "lock.acquire()\n"
    "print('acquired', flush=True)\n"
    "time.sleep(120)\n"
)
THIEF = (
    "import sys\n"
    "from qasystem.storage.lock import DataLock\n"
    "from qasystem.errors import StorageLockedError\n"
    "try:\n"
    "    DataLock(sys.argv[1]).acquire()\n"
    "    print('ACQUIRED', flush=True)\n"
    "except StorageLockedError as exc:\n"
    "    print(exc.code, flush=True)\n"
)

MODEL = "Bge-m3"
DIM = 4
VECTOR = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def store(tmp_path: Any) -> Iterator[SqliteStore]:
    instance = SqliteStore(tmp_path / "qasystem.db")
    yield instance
    instance.close()


@pytest.fixture
def vectors(tmp_path: Any) -> Iterator[ChromaStore]:
    instance = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    yield instance
    instance.close()


def embed_input(doc_id: str, text: str) -> str:
    """P3's embedding input, byte for byte: the cache is keyed on this exact string."""
    return f"{doc_id} > S\n\n{text}"


def ingest(store: SqliteStore, doc_id: str, texts: list[str]) -> int:
    """Stage and publish one document, filling the embedding cache as P6 would."""
    version = store.begin_staging(doc_id, doc_id, f"{doc_id}.md", "md", "c", "p", "en")
    chunks = [Chunk(i, 0, len(t), t, ("S",), f"hash{i}", "en") for i, t in enumerate(texts)]
    inputs = [embed_input(doc_id, t) for t in texts]
    store.stage_version(doc_id, version, "c", " ".join(texts), chunks, inputs)
    store.publish(doc_id, version, "c", "p")
    store.put_embeddings(MODEL, [(input_hash(i), VECTOR) for i in inputs])
    return version


# ---------------------------------------------------------------- plan_reconcile


def test_an_empty_system_is_in_sync(store: SqliteStore, vectors: ChromaStore) -> None:
    vectors.ensure_collection()
    plan = plan_reconcile(store, vectors)
    assert plan.in_sync is True
    assert plan.missing == ()
    assert plan.orphaned == ()


def test_a_fully_indexed_system_is_in_sync(store: SqliteStore, vectors: ChromaStore) -> None:
    ingest(store, "d", ["alpha", "beta"])
    vectors.upsert([_vector("d:v1:0"), _vector("d:v1:1")])
    assert plan_reconcile(store, vectors).in_sync is True


def test_a_missing_vector_is_reported_not_repaired(
    store: SqliteStore, vectors: ChromaStore
) -> None:
    """reconcile plans; it must never mutate SQLite based on Chroma (rule 8)."""
    ingest(store, "d", ["alpha", "beta"])
    vectors.upsert([_vector("d:v1:0")])
    plan = plan_reconcile(store, vectors)
    assert plan.missing == ("d:v1:1",)
    assert plan.in_sync is False
    assert store.expected_vector_ids() == ["d:v1:0", "d:v1:1"]  # SQLite unchanged


def test_an_orphaned_vector_is_reported(store: SqliteStore, vectors: ChromaStore) -> None:
    ingest(store, "d", ["alpha"])
    vectors.upsert([_vector("d:v1:0"), _vector("ghost:v1:0")])
    assert plan_reconcile(store, vectors).orphaned == ("ghost:v1:0",)


def test_a_staging_version_participates_in_the_plan(
    store: SqliteStore, vectors: ChromaStore
) -> None:
    version = store.begin_staging("d", "d", "d.md", "md", "c", "p", "en")
    _stage(store, "d", version, "body")
    assert plan_reconcile(store, vectors).missing == ("d:v1:0",)


def test_plans_are_sorted_so_they_are_comparable(store: SqliteStore, vectors: ChromaStore) -> None:
    ingest(store, "d", ["alpha", "beta", "gamma"])
    vectors.upsert([_vector("d:v1:0")])
    assert plan_reconcile(store, vectors).missing == ("d:v1:1", "d:v1:2")


# ---------------------------------------------------------------- I10 rebuild


def test_rebuild_restores_dense_search_with_zero_embedding_calls(
    store: SqliteStore, tmp_path: Path
) -> None:
    """I10: an index that has lost every vector is rebuilt from SQLite alone.

    The emptied index is a *second, empty* collection rather than a deleted directory,
    because Chroma caches per path for the life of the process: an in-process
    ``rmtree`` is not observable. That the operator's ``rm -rf data/chroma`` really does
    empty the index was verified out of band (a fresh process sees ``count() == 0``),
    and what this test covers is the half the code owns — restoration without the API.
    """
    ingest(store, "d", ["alpha", "beta"])
    populated = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    populated.upsert([_vector("d:v1:0"), _vector("d:v1:1")])
    assert populated.count() == 2
    populated.close()

    fresh = ChromaStore(tmp_path / "chroma-fresh", model_id=MODEL, dimension=DIM)
    assert fresh.query(VECTOR, n=2) == []
    assert rebuild(store, fresh, MODEL) == 2
    assert sorted(fresh.list_ids()) == ["d:v1:0", "d:v1:1"]
    assert {hit.id for hit in fresh.query(VECTOR, n=2)} == {"d:v1:0", "d:v1:1"}
    fresh.close()


def test_a_rebuilt_index_survives_a_restart(store: SqliteStore, tmp_path: Path) -> None:
    """L10 + I10 together: what rebuild wrote is persistent, not just in memory."""
    ingest(store, "d", ["alpha"])
    first = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    assert rebuild(store, first, MODEL) == 1
    first.close()
    second = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    assert [hit.id for hit in second.query(VECTOR, n=1)] == ["d:v1:0"]
    second.close()


def test_rebuild_does_not_call_the_embedding_api(store: SqliteStore, vectors: ChromaStore) -> None:
    """The point of rebuild: a rate-limited provider must not be involved at all."""
    ingest(store, "d", ["alpha"])
    calls = 0

    class Tripwire:
        model_id = MODEL
        dimension = DIM

        async def embed(self, texts: Any) -> Any:
            nonlocal calls
            calls += 1
            raise AssertionError("rebuild must not embed")

    assert rebuild(store, vectors, MODEL, embedder=Tripwire()) == 1
    assert calls == 0


def test_rebuild_skips_a_chunk_whose_vector_is_not_cached(
    store: SqliteStore, vectors: ChromaStore
) -> None:
    """Honest partial rebuild: report what could not be restored."""
    ingest(store, "d", ["alpha", "beta"])
    store.execute_script("DELETE FROM embedding_cache")
    assert rebuild(store, vectors, MODEL) == 0
    assert vectors.count() == 0


def test_rebuild_reports_the_uncached_ids_for_a_follow_up_call(
    store: SqliteStore, vectors: ChromaStore
) -> None:
    version = store.begin_staging("d", "d", "d.md", "md", "c", "p", "en")
    _stage(store, "d", version, "body")
    assert rebuild(store, vectors, MODEL) == 0
    assert plan_reconcile(store, vectors).missing == ("d:v1:0",)


def test_rebuild_of_an_empty_system_is_a_no_op(store: SqliteStore, vectors: ChromaStore) -> None:
    assert rebuild(store, vectors, MODEL) == 0
    assert vectors.count() == 0


def test_rebuild_is_idempotent(store: SqliteStore, vectors: ChromaStore) -> None:
    ingest(store, "d", ["alpha", "beta"])
    assert rebuild(store, vectors, MODEL) == 2
    assert rebuild(store, vectors, MODEL) == 2
    assert vectors.count() == 2
    assert plan_reconcile(store, vectors).in_sync is True


def test_rebuild_does_not_resurrect_a_purged_version(
    store: SqliteStore, vectors: ChromaStore
) -> None:
    ingest(store, "d", ["alpha"])
    store.purge_version("d", 1)
    assert rebuild(store, vectors, MODEL) == 0
    assert vectors.count() == 0


# ---------------------------------------------------------------- L2 lock


def test_the_lock_is_taken_and_released(tmp_path: Any) -> None:
    with DataLock(tmp_path):
        assert (tmp_path / ".qasystem.lock").exists()
    with DataLock(tmp_path):
        pass  # re-acquirable after release


def test_a_second_holder_is_refused(tmp_path: Any) -> None:
    """L2: two owners of one data/ directory would corrupt Chroma."""
    with DataLock(tmp_path), pytest.raises(StorageLockedError, match="another process"):
        DataLock(tmp_path).acquire()


def test_a_second_process_is_refused(tmp_path: Any) -> None:
    """L2 as it actually happens: ``uvicorn --workers 1`` is only a command-line promise.

    A same-process check is weaker than it looks, so this runs a real second process
    that holds the lock and asserts a third one is turned away. It is the only test here
    that spawns anything, and it is what stops ``--workers 2`` from corrupting Chroma.
    """
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "acquired"  # it really holds the lock
        thief = subprocess.run(
            [sys.executable, "-c", THIEF, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert thief.stdout.strip() == "STORAGE_LOCKED", thief.stdout + thief.stderr
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_the_lock_dies_with_the_process_that_held_it(tmp_path: Any) -> None:
    """A crash must not leave a directory permanently unownable."""
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(tmp_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        holder.stdout.readline()
        holder.kill()
        holder.wait(timeout=10)
    finally:
        with DataLock(tmp_path):
            pass  # acquired again, so the lock was released when the process died


def test_a_failed_acquire_leaves_no_lock_object_behind(tmp_path: Any) -> None:
    holder = DataLock(tmp_path)
    holder.acquire()
    try:
        second = DataLock(tmp_path)
        with pytest.raises(StorageLockedError):
            second.acquire()
        second.release()  # must not raise, must not release the holder's lock
        assert holder.is_held is True
    finally:
        holder.release()


def test_a_different_directory_can_be_owned_at_the_same_time(tmp_path: Any) -> None:
    with DataLock(tmp_path / "one"), DataLock(tmp_path / "two"):
        pass


def test_the_lock_creates_its_directory(tmp_path: Any) -> None:
    with DataLock(tmp_path / "deep" / "path"):
        assert (tmp_path / "deep" / "path" / ".qasystem.lock").exists()


def test_releasing_a_lock_that_was_never_taken_is_harmless(tmp_path: Any) -> None:
    DataLock(tmp_path).release()


def test_the_lock_error_has_a_stable_code(tmp_path: Any) -> None:
    assert StorageLockedError.code == "STORAGE_LOCKED"
    assert StorageLockedError.http_status == 503


def _vector(vector_id: str) -> Any:
    from qasystem.domain.models import VectorItem

    return VectorItem(id=vector_id, vector=VECTOR, metadata={"doc_id": "d", "doc_version": 1})


def _stage(store: SqliteStore, doc_id: str, version: int, text: str) -> None:
    store.stage_version(
        doc_id, version, "c", text, [Chunk(0, 0, len(text), text, ("S",), "h", "en")], [text]
    )
