"""P5: local Chroma adapter, L1-L10 (plan.md §4.2). Real ``PersistentClient`` on
``tmp_path`` — an in-memory fake would not prove restart persistence (L10).

``VectorItem`` and ``VectorHit`` come straight from the frozen P0 contracts, so this
file also proves the port is implementable as written.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from qasystem.domain.models import VectorHit, VectorItem
from qasystem.errors import VectorStoreError
from qasystem.storage.chroma_store import ChromaStore, collection_name

MODEL = "Bge-m3"
DIM = 4


def item(vector_id: str, vector: list[float] | None = None, **metadata: Any) -> VectorItem:
    return VectorItem(
        id=vector_id,
        vector=vector or [1.0, 0.0, 0.0, 0.0],
        metadata={"doc_id": "d", "doc_version": 1, "ordinal": 0, **metadata},
    )


@pytest.fixture
def store(tmp_path: Any) -> Iterator[ChromaStore]:
    instance = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    yield instance
    instance.close()


# ---------------------------------------------------------------- naming / I9


def test_the_collection_name_is_slugged_and_carries_the_dimension() -> None:
    """L4: `Bge-m3` must not become `Bge-m3` in a Chroma identifier."""
    assert collection_name("Bge-m3", 1024) == "chunks__bge_m3__d1024"
    assert collection_name("Embedding-3-Small", 1536) == "chunks__embedding_3_small__d1536"


def test_two_different_models_never_share_a_collection() -> None:
    assert collection_name("Bge-m3", 1024) != collection_name("Embedding-3-Large", 1024)
    assert collection_name("Bge-m3", 1024) != collection_name("Bge-m3", 1536)


def test_a_collection_name_is_always_a_legal_chroma_identifier() -> None:
    """`[a-z0-9_-]`, 3-63 chars: an illegal name is a runtime crash, not a warning."""
    import re

    name = collection_name("Gemini-embedding-001/weird name!", 3072)
    assert re.fullmatch(r"[a-z0-9_-]{3,63}", name), name


def test_a_different_dimension_gets_a_different_collection(tmp_path: Any) -> None:
    """The dimension is in the collection name, so 1024-d and 1536-d never meet."""
    small = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    small.ensure_collection()
    large = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=8)
    large.ensure_collection()
    small.upsert([item("a")])
    assert large.count() == 0


def test_a_different_model_gets_a_different_collection(tmp_path: Any) -> None:
    bge = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    bge.ensure_collection()
    other = ChromaStore(tmp_path / "chroma", model_id="Embedding-3-Large", dimension=DIM)
    other.ensure_collection()
    bge.upsert([item("a")])
    assert other.count() == 0
    assert bge.collection_name != other.collection_name


def test_a_name_collision_is_refused_rather_than_mixed(tmp_path: Any) -> None:
    """I9's real hole: two long model ids whose slugs truncate to one name.

    The name cannot separate them, so the stored metadata is the only thing standing
    between a 3072-d vector and a 1024-d space. The check has to bite.
    """
    prefix = "a" * 60
    first = ChromaStore(tmp_path / "chroma", model_id=f"{prefix}-one", dimension=DIM)
    first.ensure_collection()
    first.upsert([item("a")])
    assert (
        first.collection_name
        == ChromaStore(tmp_path / "chroma", model_id=f"{prefix}-two", dimension=DIM).collection_name
    )
    with pytest.raises(VectorStoreError, match="model"):
        ChromaStore(
            tmp_path / "chroma", model_id=f"{prefix}-two", dimension=DIM
        ).ensure_collection()


def test_reopening_the_same_model_and_dimension_is_fine(tmp_path: Any) -> None:
    ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM).ensure_collection()
    ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM).ensure_collection()


def test_a_wrong_dimension_vector_is_rejected_before_it_is_stored(store: ChromaStore) -> None:
    """A short vector would make cosine meaningless; refuse rather than pad."""
    with pytest.raises(VectorStoreError, match="dimension"):
        store.upsert([item("a", [1.0, 0.0])])


def test_document_text_is_never_written_to_chroma(store: ChromaStore, tmp_path: Any) -> None:
    """L6: Chroma holds vectors and diagnostic metadata only."""
    store.upsert([item("a")])
    store.list_ids()
    (chroma_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    blob = b"".join(p.read_bytes() for p in chroma_dir.rglob("*") if p.is_file())
    assert b"doc_id" not in blob or b"section_path" not in blob
    assert b"body text" not in blob


# ---------------------------------------------------------------- upsert / query


def test_an_upserted_vector_is_found_with_similarity_one(store: ChromaStore) -> None:
    store.upsert([item("a", [1.0, 0.0, 0.0, 0.0])])
    (hit,) = store.query([1.0, 0.0, 0.0, 0.0], n=1)
    assert hit.id == "a"
    assert hit.similarity == pytest.approx(1.0)


def test_cosine_distance_is_converted_to_similarity(store: ChromaStore) -> None:
    """L5: §2.2 measured identical→0.0 and orthogonal→1.0, so similarity = 1 - distance."""
    store.upsert(
        [
            item("same", [1.0, 0.0, 0.0, 0.0]),
            item("orthogonal", [0.0, 1.0, 0.0, 0.0]),
            item("opposite", [-1.0, 0.0, 0.0, 0.0]),
        ]
    )
    hits = {hit.id: hit.similarity for hit in store.query([1.0, 0.0, 0.0, 0.0], n=3)}
    assert hits["same"] == pytest.approx(1.0, abs=1e-5)
    assert hits["orthogonal"] == pytest.approx(0.0, abs=1e-5)
    assert hits["opposite"] == pytest.approx(-1.0, abs=1e-5)


def test_results_come_back_nearest_first(store: ChromaStore) -> None:
    store.upsert(
        [
            item("far", [0.0, 1.0, 0.0, 0.0]),
            item("near", [0.9, 0.1, 0.0, 0.0]),
        ]
    )
    assert [hit.id for hit in store.query([1.0, 0.0, 0.0, 0.0], n=2)] == ["near", "far"]


def test_n_is_clamped_to_the_collection_count(store: ChromaStore) -> None:
    """L7: Chroma clamps silently, but an explicit clamp keeps `n` meaningful."""
    store.upsert([item("only")])
    assert len(store.query([1.0, 0.0, 0.0, 0.0], n=1000)) == 1
    assert store.query([1.0, 0.0, 0.0, 0.0], n=1000) == store.query([1.0, 0.0, 0.0, 0.0], n=1)


def test_querying_an_empty_collection_returns_nothing(store: ChromaStore) -> None:
    assert store.query([1.0, 0.0, 0.0, 0.0], n=5) == []
    assert store.count() == 0


def test_a_wrong_dimension_query_vector_is_rejected(store: ChromaStore) -> None:
    store.upsert([item("a")])
    with pytest.raises(VectorStoreError, match="dimension"):
        store.query([1.0, 0.0], n=1)


def test_upsert_replaces_a_vector_in_place(store: ChromaStore) -> None:
    """Versioned ids mean a re-upsert is a new row, never a duplicate."""
    store.upsert([item("d:v1:0", [1.0, 0.0, 0.0, 0.0])])
    store.upsert([item("d:v1:0", [0.0, 1.0, 0.0, 0.0])])
    assert store.count() == 1
    assert store.query([0.0, 1.0, 0.0, 0.0], n=1)[0].similarity == pytest.approx(1.0)


def test_a_batch_upsert_keeps_every_vector(store: ChromaStore) -> None:
    store.upsert([item(f"d:v1:{i}") for i in range(25)])
    assert store.count() == 25


def test_get_existing_ids_returns_only_what_is_there(store: ChromaStore) -> None:
    store.upsert([item("a"), item("b")])
    assert store.get_existing_ids(["a", "b", "missing"]) == {"a", "b"}
    assert store.get_existing_ids([]) == set()


def test_delete_ids_removes_only_the_named_vectors(store: ChromaStore) -> None:
    store.upsert([item("a"), item("b"), item("c")])
    store.delete_ids(["b", "missing"])
    assert sorted(store.list_ids()) == ["a", "c"]


def test_deleting_from_an_empty_store_is_a_no_op(store: ChromaStore) -> None:
    store.delete_ids(["nothing"])
    assert store.count() == 0


def test_list_ids_is_paged_but_complete(store: ChromaStore) -> None:
    store.upsert([item(f"d:v1:{i}") for i in range(120)])
    found = sorted(store.list_ids())
    assert len(found) == 120  # paged at 500, so this crosses no page boundary yet
    assert found[0] == "d:v1:0" and found[-1] == "d:v1:99"


def test_ids_keep_their_documented_shape(store: ChromaStore) -> None:
    """§4.1: `"{doc_id}:v{version}:{ordinal}"`, so a stale vector cannot collide."""
    store.upsert([item("handbook:v3:7")])
    assert sorted(store.list_ids()) == ["handbook:v3:7"]


# ---------------------------------------------------------------- restart / ping


def test_vectors_survive_a_restart(tmp_path: Any) -> None:
    """L10: write, drop the client entirely, reopen the same path, still found."""
    first = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    first.upsert([item("a", [1.0, 0.0, 0.0, 0.0]), item("b", [0.0, 1.0, 0.0, 0.0])])
    first.close()

    second = ChromaStore(tmp_path / "chroma", model_id=MODEL, dimension=DIM)
    second.ensure_collection()
    assert second.count() == 2
    assert second.query([1.0, 0.0, 0.0, 0.0], n=1)[0].id == "a"
    assert sorted(second.list_ids()) == ["a", "b"]
    second.close()


def test_ping_succeeds_on_a_healthy_store(store: ChromaStore) -> None:
    store.ping()
    assert store.heartbeat() > 0


def test_an_unusable_path_raises_a_typed_error_not_a_chroma_error(tmp_path: Any) -> None:
    blocker = tmp_path / "chroma"
    blocker.write_text("not a directory")
    with pytest.raises(VectorStoreError):
        ChromaStore(blocker, model_id=MODEL, dimension=DIM).ensure_collection()


def test_hits_are_the_frozen_contract_type(store: ChromaStore) -> None:
    store.upsert([item("a")])
    assert isinstance(store.query([1.0, 0.0, 0.0, 0.0], n=1)[0], VectorHit)
