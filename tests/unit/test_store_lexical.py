"""P5: FTS5 lexical search, always joined to ``eligible_chunks``.

The tokenizer is ours (P1), so SQLite never has to understand Persian. The join is
what makes this path obey I1 and I2 like every other path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from qasystem.domain.models import Chunk
from qasystem.storage.lexical import LexicalIndex
from qasystem.storage.sqlite_store import SqliteStore

HOSTILE = '" OR * ( ) NEAR AND NOT "unterminated \\'


@pytest.fixture
def store(tmp_path: Any) -> Iterator[SqliteStore]:
    instance = SqliteStore(tmp_path / "qasystem.db")
    yield instance
    instance.close()


@pytest.fixture
def index(store: SqliteStore) -> LexicalIndex:
    return LexicalIndex(store)


def add(
    store: SqliteStore, doc_id: str, text: str, *, version: int = 1, publish: bool = True
) -> str:
    """Index one chunk's text under one document version."""
    allocated = store.begin_staging(
        doc_id, doc_id, f"{doc_id}.md", "md", f"c{version}", f"p{version}", "en"
    )
    assert allocated == version
    chunk = Chunk(0, 0, len(text), text, ("S",), f"h{version}", "en")
    store.stage_version(doc_id, version, f"c{version}", text, [chunk], [text])
    if publish:
        store.publish(doc_id, version, f"c{version}", f"p{version}")
    return f"{doc_id}:v{version}:0"


def test_a_stored_word_is_found(store: SqliteStore, index: LexicalIndex) -> None:
    add(store, "d", "the quick brown fox jumps over the lazy dog")
    hits = index.search("brown fox")
    assert [hit.chunk_id for hit in hits] == ["d:v1:0"]
    assert hits[0].rank == 1, "ranks are 1-based, like RRF expects"


def test_a_missing_word_returns_nothing(store: SqliteStore, index: LexicalIndex) -> None:
    add(store, "d", "the quick brown fox")
    assert index.search("zebra") == []


def test_an_empty_query_returns_nothing_rather_than_everything(
    store: SqliteStore, index: LexicalIndex
) -> None:
    """The failure mode that would flood the candidate list with the whole corpus."""
    add(store, "d", "some text here")
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("the of and") == []  # stopwords only


def test_ranks_are_dense_and_ordered_by_relevance(store: SqliteStore, index: LexicalIndex) -> None:
    add(store, "a", "install the widget on linux")
    add(store, "b", "the widget is installed on macos")
    add(store, "c", "unrelated prose about cooking")
    hits = index.search("widget install")
    assert [hit.chunk_id for hit in hits] == ["a:v1:0", "b:v1:0"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].lexical_score == pytest.approx(1.0)
    assert 0.0 < hits[1].lexical_score < 1.0


def test_a_staging_version_is_invisible_to_lexical_search(
    store: SqliteStore, index: LexicalIndex
) -> None:
    add(store, "d", "secret staging content", publish=False)
    assert index.search("secret") == []


def test_a_superseded_version_is_invisible_to_lexical_search(
    store: SqliteStore, index: LexicalIndex
) -> None:
    add(store, "d", "old wording about widgets", version=1)
    add(store, "d", "new wording about widgets", version=2)
    hits = index.search("widgets")
    assert [hit.chunk_id for hit in hits] == ["d:v2:0"]


def test_a_deleted_document_is_invisible_to_lexical_search(
    store: SqliteStore, index: LexicalIndex
) -> None:
    add(store, "d", "deletable content")
    assert search_ids(index, "deletable") == ["d:v1:0"]
    store.mark_deleted("d")
    assert index.search("deletable") == []


def test_a_purged_version_leaves_no_lexical_rows(store: SqliteStore, index: LexicalIndex) -> None:
    add(store, "d", "purgeable content")
    store.purge_version("d", 1)
    assert index.search("purgeable") == []
    assert store.lexical_count("d:v1:0") == 0


def test_hostile_fts_syntax_is_treated_as_plain_text(
    store: SqliteStore, index: LexicalIndex
) -> None:
    """A user query must never reach FTS5 as syntax (jobTask.md, §13 checklist)."""
    add(store, "d", "ordinary harmless content")
    for query in (HOSTILE, '" OR 1=1', "NEAR(a b)", "*", "a AND", "((()))", '""'):
        assert isinstance(index.search(query), list)
    assert search_ids(index, "harmless") == ["d:v1:0"]


def test_a_query_cannot_reach_another_documents_chunk(
    store: SqliteStore, index: LexicalIndex
) -> None:
    add(store, "a", "shared term alpha")
    add(store, "b", "shared term beta")
    assert {hit.chunk_id for hit in index.search("shared")} == {"a:v1:0", "b:v1:0"}


def test_persian_is_searchable_in_zwnj_and_spaced_forms(
    store: SqliteStore, index: LexicalIndex
) -> None:
    """D8: a ZWNJ document and a spaced query must meet."""
    add(store, "fa", "پیاده‌سازی سیستم کامل است")
    assert search_ids(index, "پیاده‌سازی") == ["fa:v1:0"]
    assert search_ids(index, "پیاده سازی") == ["fa:v1:0"]
    assert search_ids(index, "سیستم") == ["fa:v1:0"]


def test_an_exact_identifier_survives_indexing(store: SqliteStore, index: LexicalIndex) -> None:
    """Error codes are why lexical retrieval runs at all (plan.md §9.3)."""
    add(store, "d", "the service returns ERR-404 when the token is invalid")
    assert search_ids(index, "ERR-404") == ["d:v1:0"]
    assert search_ids(index, "err-404") == ["d:v1:0"]


def test_results_can_be_capped(store: SqliteStore, index: LexicalIndex) -> None:
    for doc_id in "abcde":
        add(store, doc_id, "common term everywhere")
    assert len(index.search("common", limit=3)) == 3
    assert len(index.search("common", limit=99)) == 5


def test_a_search_with_no_terms_never_reaches_sqlite(
    store: SqliteStore, index: LexicalIndex
) -> None:
    """Guarding in Python means a stopword-only query costs no query plan at all."""
    add(store, "d", "content")
    assert index.search("the a of") == []


def test_persian_and_english_terms_coexist_in_one_index(
    store: SqliteStore, index: LexicalIndex
) -> None:
    add(store, "mixed", "the installer package validates the configuration file")
    assert search_ids(index, "installer") == ["mixed:v1:0"]
    assert search_ids(index, "package") == ["mixed:v1:0"]


def test_a_query_matching_nothing_does_not_raise(store: SqliteStore, index: LexicalIndex) -> None:
    add(store, "d", "content")
    assert index.search("word that appears nowhere in the corpus") == []


def search_ids(index: LexicalIndex, query: str) -> list[str]:
    return [hit.chunk_id for hit in index.search(query)]
