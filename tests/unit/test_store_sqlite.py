"""P5: schema, transactions, and the ``eligible_chunks`` choke point.

I1 and I2 are enforced by one view, so these tests attack the view rather than the
application: a row is retrievable if and only if it survives that join.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest

from qasystem.domain.models import Chunk
from qasystem.embeddings.caching import input_hash
from qasystem.storage.sqlite_store import SqliteStore

SCHEMA_VERSION = 1


@pytest.fixture
def store(tmp_path: Any) -> Iterator[SqliteStore]:
    instance = SqliteStore(tmp_path / "qasystem.db")
    yield instance
    instance.close()


def make_chunk(ordinal: int = 0, text: str = "body text here") -> Chunk:
    return Chunk(
        ordinal=ordinal,
        char_start=0,
        char_end=len(text),
        text=text,
        section_path=("Guide", "Install"),
        chunk_hash=f"hash{ordinal}",
        language="en",
    )


def published(store: SqliteStore, doc_id: str, *, text: str = "body", version: int = 1) -> None:
    """Put one document at one published version, the way P6 would."""
    allocated = store.begin_staging(
        doc_id, doc_id, f"{doc_id}.md", "md", f"c{version}", f"p{version}", "en"
    )
    assert allocated == version
    store.stage_version(doc_id, version, f"c{version}", text, [make_chunk(0, text)], [text])
    store.publish(doc_id, version, f"c{version}", f"p{version}")


# ---------------------------------------------------------------- schema


def test_creating_the_schema_twice_is_a_no_op(tmp_path: Any) -> None:
    first = SqliteStore(tmp_path / "a.db")
    published(first, "d")
    first.close()
    second = SqliteStore(tmp_path / "a.db")
    assert second.documents() == ["d"]  # data survived, schema not rebuilt
    assert second.schema_version() == SCHEMA_VERSION
    second.close()


def test_the_database_file_and_its_parent_are_created(tmp_path: Any) -> None:
    path = tmp_path / "nested" / "deeper" / "qasystem.db"
    store = SqliteStore(path)
    assert path.exists()
    store.close()


def test_wal_and_foreign_keys_are_on(tmp_path: Any) -> None:
    store = SqliteStore(tmp_path / "q.db")
    assert store.journal_mode() == "wal"
    assert store.foreign_keys_enabled() is True
    store.close()


def test_this_sqlite_build_has_fts5() -> None:
    """plan.md §6: FTS5 presence is a startup check, not an assumption."""
    assert SqliteStore.fts5_available()


# ---------------------------------------------------------------- transactions


def test_a_failed_write_leaves_no_partial_rows(store: SqliteStore) -> None:
    """I3's foundation: staging is all-or-nothing, so a failure leaves the old version alone."""
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    with pytest.raises(RuntimeError, match="injected"):
        store.stage_version(
            doc_id="d",
            version=version,
            content_hash="c1",
            source_text="the whole document",
            chunks=[make_chunk(0), make_chunk(1)],
            embed_inputs=["a", "b"],
            fail_after="chunks",
        )
    assert store.chunk_count("d", version) == 0
    assert store.versions("d") == []
    assert store.lexical_count("d:v1:0") == 0


def test_rollback_leaves_the_previous_published_version_untouched(store: SqliteStore) -> None:
    published(store, "d", text="v1 body", version=1)
    version = store.begin_staging("d", "D", "d.md", "md", "c2", "p2", "en")
    with pytest.raises(RuntimeError):
        store.stage_version(
            "d", version, "c2", "v2 body", [make_chunk(0, "v2 body")], ["v2 body"], fail_after="fts"
        )
    assert store.eligible_chunk_ids() == ["d:v1:0"]
    assert store.source_text("d", 1) == "v1 body"
    assert store.version_state("d", 1) == "published"


def test_a_rolled_back_staging_can_be_retried(store: SqliteStore) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    with pytest.raises(RuntimeError):
        store.stage_version("d", version, "c1", "text", [make_chunk(0)], ["t"], fail_after="chunks")
    store.stage_version("d", version, "c1", "text", [make_chunk(0)], ["t"])
    assert store.chunk_count("d", version) == 1


def test_publishing_an_unstaged_version_changes_nothing(store: SqliteStore) -> None:
    store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    with pytest.raises(sqlite3.IntegrityError):
        store.publish("d", 7, "c7", "p7")
    assert store.get_document("d")["current_version"] is None
    assert store.eligible_chunk_ids() == []


# ---------------------------------------------------------------- documents


def test_a_new_document_has_no_current_version_until_publish(store: SqliteStore) -> None:
    store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    row = store.get_document("d")
    assert row["current_version"] is None
    assert row["last_version"] == 1  # already allocated by begin_staging


def test_version_numbers_are_allocated_monotonically(store: SqliteStore) -> None:
    assert store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en") == 1
    assert store.begin_staging("d", "D", "d.md", "md", "c2", "p2", "en") == 2


def test_a_version_number_is_never_reused_after_publish(store: SqliteStore) -> None:
    published(store, "d", version=1)
    published(store, "d", version=2)
    assert store.begin_staging("d", "D", "d.md", "md", "c3", "p3", "en") == 3


def test_a_deleted_document_keeps_its_version_counter(store: SqliteStore) -> None:
    published(store, "d", version=1)
    store.mark_deleted("d")
    assert store.get_document("d")["status"] == "deleted"
    assert store.begin_staging("d", "D", "d.md", "md", "c2", "p2", "en") == 2


def test_re_adding_a_deleted_document_makes_it_active_again(store: SqliteStore) -> None:
    published(store, "d", version=1)
    store.mark_deleted("d")
    version = store.begin_staging("d", "D", "d.md", "md", "c2", "p2", "en")
    store.stage_version("d", version, "c2", "new body", [make_chunk(0, "new body")], ["body"])
    store.publish("d", version, "c2", "p2")
    assert store.get_document("d")["status"] == "active"
    assert store.eligible_chunk_ids() == ["d:v2:0"]


def test_an_unknown_document_is_none_not_an_error(store: SqliteStore) -> None:
    assert store.get_document("nope") is None
    assert store.get_document("missing") is None
    assert store.documents() == []


# ---------------------------------------------------------------- I1 / I2


def test_a_staging_chunk_is_not_eligible(store: SqliteStore) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version("d", version, "c1", "body", [make_chunk(0)], ["body"])
    assert store.eligible_chunk_ids() == []


def test_a_published_chunk_is_eligible(store: SqliteStore) -> None:
    published(store, "d")
    assert store.eligible_chunk_ids() == ["d:v1:0"]


def test_a_superseded_chunk_is_not_eligible(store: SqliteStore) -> None:
    """The I1/I2 core: publishing v2 must make v1 unreachable, rows and all."""
    published(store, "d", text="v1 body", version=1)
    published(store, "d", text="v2 body", version=2)
    assert store.eligible_chunk_ids() == ["d:v2:0"]


def test_a_deleted_document_has_no_eligible_chunks(store: SqliteStore) -> None:
    published(store, "d")
    assert store.eligible_chunk_ids() == ["d:v1:0"]
    store.mark_deleted("d")
    assert store.eligible_chunk_ids() == []


def test_a_deleted_document_is_ineligible_even_with_a_stale_current_version(
    store: SqliteStore,
) -> None:
    """I2 has two guards in the view, and this pins the one ``mark_deleted`` hides.

    ``mark_deleted`` nulls ``current_version`` *and* flips ``status``, so testing only
    the method leaves the ``status`` predicate unexercised — a mutation that deletes
    ``d.status = 'active'`` from the view passes every other test. Setting the column
    directly reproduces the half-applied delete that the second guard exists for.
    """
    published(store, "d")
    store.execute_script("UPDATE documents SET status = 'deleted'")
    assert store.get_document("d")["current_version"] == 1  # the dangerous state
    assert store.eligible_chunk_ids() == []


def test_another_documents_publish_does_not_hide_this_one(store: SqliteStore) -> None:
    published(store, "a")
    published(store, "b")
    assert store.eligible_chunk_ids() == ["a:v1:0", "b:v1:0"]


def test_publish_sets_the_document_current_version_and_hashes(store: SqliteStore) -> None:
    published(store, "d", version=1)
    row = store.get_document("d")
    assert row["current_version"] == 1
    assert row["content_hash"] == "c1"
    assert row["parsed_hash"] == "p1"
    assert row["status"] == "active"
    assert row["published_at"] is not None
    assert row["deleted_at"] is None


def test_publishing_marks_the_previous_version_superseded(store: SqliteStore) -> None:
    published(store, "d", text="v1", version=1)
    published(store, "d", text="v2", version=2)
    assert store.version_state("d", 1) == "superseded"
    assert store.version_state("d", 2) == "published"


def test_source_text_is_retained_for_I6_verification(store: SqliteStore) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version("d", version, "c1", "exact source text", [make_chunk(0)], ["body"])
    assert store.source_text("d", version) == "exact source text"
    assert store.source_text("d", 99) is None


def test_a_stored_chunk_slices_its_own_source_text(store: SqliteStore) -> None:
    """I6, checked in storage rather than trusted from the chunker."""
    text = "First paragraph. Second paragraph."
    chunk = Chunk(
        ordinal=0,
        char_start=18,
        char_end=len(text),
        text=text[18:],
        section_path=("S",),
        chunk_hash="h",
        language="en",
    )
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version("d", version, "c1", text, [chunk], [text])
    store.publish("d", version, "c1", "p1")
    stored = store.eligible_chunks(chunk_ids=["d:v1:0"])[0]
    assert stored["text"] == text[18:]


def test_eligible_lookup_filters_by_document_and_language(store: SqliteStore) -> None:
    for doc_id, language in (("en-doc", "en"), ("fa-doc", "fa")):
        version = store.begin_staging(doc_id, doc_id, f"{doc_id}.md", "md", "c", "p", language)
        chunk = Chunk(0, 0, 4, "body", ("S",), "h", language)  # type: ignore[arg-type]
        store.stage_version(doc_id, version, "c", "body", [chunk], ["body"])
        store.publish(doc_id, version, "c", "p")
    assert [row["doc_id"] for row in store.eligible_chunks(doc_ids=["fa-doc"])] == ["fa-doc"]
    assert [row["doc_id"] for row in store.eligible_chunks(language="fa")] == ["fa-doc"]
    assert len(store.eligible_chunks()) == 2
    assert store.eligible_chunks(chunk_ids=["nope:v1:0"]) == []


def test_expected_vector_ids_covers_every_chunk_row(store: SqliteStore) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version("d", version, "c1", "body", [make_chunk(0), make_chunk(1)], ["a", "b"])
    store.publish("d", version, "c1", "p1")
    assert store.expected_vector_ids() == ["d:v1:0", "d:v1:1"]


def test_expected_vector_ids_include_staging_rows_so_reconcile_sees_them(
    store: SqliteStore,
) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version("d", version, "c1", "body", [make_chunk(0)], ["body"])
    assert store.expected_vector_ids() == ["d:v1:0"]


def test_chunks_for_version_returns_what_a_diff_needs(store: SqliteStore) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version(
        "d", version, "c1", "body", [make_chunk(0, "aaa"), make_chunk(1, "bbb")], ["a", "b"]
    )
    rows = store.chunks_for_version("d", version)
    assert [row["chunk_hash"] for row in rows] == ["hash0", "hash1"]
    assert [row["embed_input_hash"] for row in rows] == [input_hash("a"), input_hash("b")]


# ---------------------------------------------------------------- I5 cache


def test_a_cached_vector_round_trips_exactly(store: SqliteStore) -> None:
    vector = [0.123456789, -0.987654321, 0.5, 0.0]
    store.put_embeddings("Bge-m3", [("abc123", vector)])
    assert store.get_embeddings("Bge-m3", ["abc123"])["abc123"] == pytest.approx(vector, abs=1e-6)


def test_the_cache_is_keyed_by_model_too(store: SqliteStore) -> None:
    store.put_embeddings("Bge-m3", [("h", [1.0, 0.0])])
    store.put_embeddings("other", [("h", [0.0, 1.0])])
    assert store.get_embeddings("Bge-m3", ["h"])["h"] == pytest.approx([1.0, 0.0])
    assert store.get_embeddings("other", ["h"])["h"] == pytest.approx([0.0, 1.0])


def test_a_cache_miss_returns_nothing_rather_than_zeroes(store: SqliteStore) -> None:
    """A zero vector would be a plausible-looking but wrong answer (I9)."""
    assert store.get_embeddings("Bge-m3", ["missing"]) == {}
    assert store.get_embeddings("Bge-m3", []) == {}


def test_cached_dimensions_are_recorded(store: SqliteStore) -> None:
    store.put_embeddings("Bge-m3", [("h", [1.0, 0.0, 0.0])])
    assert store.cached_dimensions("Bge-m3") == {3}


def test_rewriting_a_cache_row_is_an_upsert(store: SqliteStore) -> None:
    store.put_embeddings("Bge-m3", [("h", [1.0, 0.0])])
    store.put_embeddings("Bge-m3", [("h", [0.0, 1.0])])
    assert store.get_embeddings("Bge-m3", ["h"])["h"] == pytest.approx([0.0, 1.0])
    assert store.cached_dimensions("Bge-m3") == {2}


# ---------------------------------------------------------------- log + cleanup


def test_ingest_log_records_counts_but_never_text(store: SqliteStore) -> None:
    store.log_ingest(
        doc_id="d",
        action="publish",
        prev_version=None,
        new_version=1,
        chunks_added=3,
        chunks_reused=7,
        embed_requests=3,
        duration_ms=42,
        status="ok",
    )
    row = store.ingest_log("d")[0]
    assert (row["chunks_added"], row["chunks_reused"], row["embed_requests"]) == (3, 7, 3)
    assert row["duration_ms"] == 42
    assert "body text here" not in str(row)


def test_ingest_log_can_record_a_failure_code(store: SqliteStore) -> None:
    store.log_ingest("d", "publish", None, 2, error_code="EMBEDDING_UNAVAILABLE", status="failed")
    assert store.ingest_log("d")[0]["error_code"] == "EMBEDDING_UNAVAILABLE"


def test_purge_removes_a_versions_rows_but_keeps_the_document(store: SqliteStore) -> None:
    published(store, "d", text="v1", version=1)
    version = store.begin_staging("d", "D", "d.md", "md", "c2", "p2", "en")
    store.stage_version("d", version, "c2", "v2", [make_chunk(0, "v2")], ["v2"])
    store.purge_version("d", version)
    assert store.chunk_count("d", version) == 0
    assert store.lexical_count("d:v2:0") == 0
    assert store.documents() == ["d"]  # the tombstone stays


def test_purging_a_published_version_makes_its_evidence_unreachable(store: SqliteStore) -> None:
    """Cleanup is best-effort, so the view must hold even if it never runs (I2)."""
    published(store, "d")
    assert store.eligible_chunk_ids() == ["d:v1:0"]
    store.purge_version("d", 1)
    assert store.eligible_chunk_ids() == []


def test_a_failed_version_is_marked_and_purgeable(store: SqliteStore) -> None:
    version = store.begin_staging("d", "D", "d.md", "md", "c1", "p1", "en")
    store.stage_version("d", version, "c1", "body", [make_chunk(0)], ["body"])
    store.mark_failed("d", version)
    assert store.version_state("d", version) == "failed"
    assert store.eligible_chunk_ids() == []


def test_the_raw_connection_is_not_leaked_to_callers(store: SqliteStore) -> None:
    """Repository methods own the connection; a leaked cursor would bypass them."""
    store.execute_script("SELECT 1")
    assert store.in_transaction() is False
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.execute_script("SELECT 1")
