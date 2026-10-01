"""P6: ingestion and change management -- the twelve tests the plan's Gate names.

Real SQLite, a real ``PersistentClient`` on ``tmp_path``, and a deterministic fake
embedder. Faults are injected by wrapping the real objects, never by mocking them, so a
test failure means the production path broke rather than that a mock drifted.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

from qasystem.chunking.chunker import Chunker
from qasystem.domain.models import VectorItem
from qasystem.embeddings.caching import CachingEmbedder
from qasystem.embeddings.fake import FakeEmbedder
from qasystem.errors import DocumentNotFoundError, EmbeddingUnavailableError, VectorStoreError
from qasystem.ingestion.service import IngestionService, default_doc_id
from qasystem.parsing.registry import ParserRegistry
from qasystem.storage.chroma_store import ChromaStore
from qasystem.storage.lexical import LexicalIndex
from qasystem.storage.sqlite_store import SqliteStore

DIM = 8

DOC_V1 = """# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-404 when the token is invalid.
"""

DOC_V1_EDITED = """# Handbook

## Install

Run the installer on Linux. It checks the kernel version first.

## Usage

The service returns ERR-503 when the upstream is unreachable.
"""

DOC_V2_EXTRA = (
    DOC_V1
    + """
## Limits

The upload ceiling is twenty megabytes.
"""
)


class FlakyVectors:
    """Wraps a real ``ChromaStore`` so a named method can be made to fail on demand.

    The point is to inject a *storage* fault at the point P6 depends on it not mattering,
    while everything else stays the production adapter.
    """

    def __init__(self, inner: ChromaStore) -> None:
        self._inner = inner
        self.fail_on: str | None = None

    def _maybe_fail(self, name: str) -> None:
        if self.fail_on == name:
            raise VectorStoreError(f"injected {name} failure")

    def ensure_collection(self) -> None:
        self._maybe_fail("ensure_collection")
        self._inner.ensure_collection()

    def upsert(self, items: Sequence[VectorItem]) -> None:
        self._maybe_fail("upsert")
        self._inner.upsert(items)

    def delete_ids(self, ids: Sequence[str]) -> None:
        self._maybe_fail("delete_ids")
        self._inner.delete_ids(ids)

    def query(self, vector: Sequence[float], n: int) -> list[Any]:
        return self._inner.query(vector, n)

    def get_existing_ids(self, ids: Sequence[str]) -> set[str]:
        return self._inner.get_existing_ids(ids)

    def list_ids(self) -> Iterator[str]:
        return self._inner.list_ids()

    def count(self) -> int:
        return self._inner.count()

    def ping(self) -> None:
        self._inner.ping()


class ExplodingEmbedder:
    """Fails on the Nth call, so a mid-update failure is reproducible."""

    def __init__(self, inner: FakeEmbedder, *, on_call: int) -> None:
        self._inner = inner
        self._on_call = on_call
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def texts_embedded(self) -> int:
        return self._inner.texts_embedded

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls == self._on_call:
            raise EmbeddingUnavailableError("injected embedder failure")
        return await self._inner.embed(texts)


@pytest.fixture
def env(tmp_path: Any, settings_factory: Any) -> Any:
    """A complete, isolated ingestion environment: real storage, fake embeddings."""
    settings = settings_factory(app_env="test", embedding_provider="fake", embedding_dimension=DIM)
    store = SqliteStore(tmp_path / "qasystem.db")
    inner = ChromaStore(tmp_path / "chroma", model_id="fake-embedder", dimension=DIM)
    vectors = FlakyVectors(inner)
    fake = FakeEmbedder(dimension=DIM, model_id="fake-embedder")
    embedder = CachingEmbedder(fake, _Cache(store))

    class Env:
        def __init__(self) -> None:
            self.settings = settings
            self.store = store
            self.vectors = vectors
            self.real = inner
            self.fake = fake
            self.embedder = embedder
            self.index = LexicalIndex(store)
            self.service = IngestionService(
                store=store,
                vectors=vectors,  # type: ignore[arg-type]
                embedder=embedder,
                parser=ParserRegistry(),
                chunker=Chunker(settings),
            )

        def eligible(self) -> list[str]:
            return self.store.eligible_chunk_ids()

        def dense_ids(self) -> list[str]:
            return sorted(self.real.list_ids())

        def chroma_readable_ids(self) -> set[str]:
            """The join P7's dense path performs, so I2 is provable at this layer."""
            return {
                row["chunk_id"] for row in self.store.eligible_chunks(chunk_ids=self.dense_ids())
            }

        def lexical_ids(self, term: str) -> list[str]:
            return [hit.chunk_id for hit in self.index.search(term)]

    yield Env()

    store.close()
    inner.close()


class _Cache:
    """``EmbeddingCache`` over the store, so I5 and I10 use the real table."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def get_many(self, model_id: str, input_hashes: Sequence[str]) -> dict[str, list[float]]:
        return self._store.get_embeddings(model_id, input_hashes)

    def put_many(self, model_id: str, rows: Any) -> None:
        self._store.put_embeddings(model_id, rows.items())


# ---------------------------------------------------------------- test 1


async def test_1_add_then_query_finds_the_content(env: Any) -> None:
    result = await env.service.ingest(DOC_V1.encode(), "handbook.md")
    assert result.status == "created"
    assert result.version == 1
    assert result.chunks_added > 0
    assert env.eligible() == [f"handbook:v1:{i}" for i in range(result.chunks_added)]
    assert env.lexical_ids("installer")
    assert default_doc_id("handbook.md") == "handbook"


# ---------------------------------------------------------------- test 2 (I4)


async def test_2_identical_reupload_is_unchanged_and_costs_zero_embed_calls(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    env.fake.reset()

    again = await env.service.ingest(DOC_V1.encode(), "handbook.md")
    assert again.status == "unchanged"
    assert again.version == 1
    assert again.embed_requests == 0
    assert env.fake.texts_embedded == 0
    assert len(env.store.versions("handbook")) == 1  # no new version


async def test_2b_different_bytes_with_identical_parsed_text_are_also_unchanged(env: Any) -> None:
    """I4 says *parsed text*, not raw bytes: a CRLF re-upload is not a new version."""
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    env.fake.reset()

    crlf = DOC_V1.replace("\n", "\r\n")
    assert crlf != DOC_V1 and crlf.encode() != DOC_V1.encode()
    again = await env.service.ingest(crlf.encode(), "handbook.md")
    assert again.status == "unchanged"
    assert env.fake.texts_embedded == 0
    assert len(env.store.versions("handbook")) == 1


# ---------------------------------------------------------------- test 3


async def test_3_a_local_edit_embeds_only_the_changed_chunk(env: Any) -> None:
    first = await env.service.ingest(DOC_V1.encode(), "handbook.md")
    env.fake.reset()

    second = await env.service.ingest(DOC_V1_EDITED.encode(), "handbook.md")
    assert second.status == "updated"
    assert second.version == 2
    assert second.chunks_added == 1
    assert second.chunks_reused == first.chunks_added - 1
    # Exactly one chunk's worth of text was embedded.
    assert 0 < env.fake.texts_embedded <= first.chunks_added
    assert second.embed_requests == 1


# ---------------------------------------------------------------- test 4 (I1)


async def test_4_after_publish_the_old_text_is_unretrievable(env: Any) -> None:
    first = await env.service.ingest(DOC_V1.encode(), "handbook.md")
    assert env.lexical_ids("ERR-404")

    await env.service.ingest(DOC_V1_EDITED.encode(), "handbook.md")
    assert env.lexical_ids("ERR-404") == [], "superseded text is still lexially reachable"
    assert env.lexical_ids("ERR-503")
    assert all(":v2:" in i for i in env.eligible())
    assert len(env.eligible()) == first.chunks_added, "the new version is whole"

    # Correctness above does not depend on cleanup -- the view already refuses v1 -- so
    # assert cleanup happened too. Otherwise deleting it would be invisible, which is
    # exactly what a mutation check caught (D34).
    assert env.store.chunk_count("handbook", 1) == 0, "superseded rows were kept"
    assert env.store.version_state("handbook", 1) is None
    assert not any(":v1:" in vector_id for vector_id in env.dense_ids())
    assert env.store.ingest_log("handbook")[-1]["chunks_removed"] == 1


# ---------------------------------------------------------------- test 5 (I2)


async def test_5_delete_is_unretrievable_even_when_chroma_cleanup_fails(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    await env.service.ingest(DOC_V2_EXTRA.encode(), "handbook.md")
    stale = env.dense_ids()
    assert stale, "precondition: vectors exist"

    env.vectors.fail_on = "delete_ids"
    await env.service.delete("handbook")  # must not raise

    assert env.eligible() == []
    assert env.lexical_ids("installer") == []
    # Chroma still holds the vectors, and the join still refuses every one of them.
    assert env.dense_ids() == stale
    assert env.chroma_readable_ids() == set()


async def test_5b_delete_purges_chroma_when_cleanup_succeeds(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    await env.service.delete("handbook")
    assert env.dense_ids() == []
    row = env.store.get_document("handbook")
    # Both halves of the delete are asserted on the row itself. Testing only the view is
    # what let a mutation that drops current_version = NULL pass every test (D26): the
    # status='active' guard alone already excludes the document.
    assert row["status"] == "deleted"
    assert row["current_version"] is None, "schema says current_version is NULL while deleted"
    assert row["deleted_at"] is not None
    assert env.store.chunk_count("handbook", 1) == 0
    assert env.store.lexical_count("handbook:v1:0") == 0


async def test_5c_deleting_an_unknown_document_raises(env: Any) -> None:
    with pytest.raises(DocumentNotFoundError):
        await env.service.delete("never-existed")


# ---------------------------------------------------------------- test 6


async def test_6_re_add_creates_a_new_version_and_old_ids_never_return(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    first_ids = env.dense_ids()
    await env.service.delete("handbook")

    again = await env.service.ingest(DOC_V2_EXTRA.encode(), "handbook.md")
    assert again.version == 2, "version numbers are never reused, even across delete"
    now = env.dense_ids()
    assert not set(first_ids) & set(now)
    assert env.lexical_ids("twenty megabytes")
    assert env.store.get_document("handbook")["status"] == "active"


# ---------------------------------------------------------------- test 7 (I3)


async def test_7_embedder_failure_mid_update_leaves_the_old_version_queryable(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    before_lexical = env.lexical_ids("installer")
    before_eligible = env.eligible()

    exploding = ExplodingEmbedder(FakeEmbedder(dimension=DIM), on_call=1)
    service = IngestionService(
        store=env.store,
        vectors=env.vectors,  # type: ignore[arg-type]
        embedder=CachingEmbedder(exploding, _Cache(env.store)),
        parser=ParserRegistry(),
        chunker=Chunker(env.settings),
    )
    with pytest.raises(EmbeddingUnavailableError):
        await service.ingest(DOC_V2_EXTRA.encode(), "handbook.md")

    assert env.eligible() == before_eligible
    assert env.lexical_ids("installer") == before_lexical
    assert env.lexical_ids("twenty megabytes") == [], "the failed version leaked"
    assert env.store.get_document("handbook")["current_version"] == 1


async def test_7b_a_failed_version_is_marked_and_leaves_no_rows(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    exploding = ExplodingEmbedder(FakeEmbedder(dimension=DIM), on_call=1)
    service = IngestionService(
        store=env.store,
        vectors=env.vectors,  # type: ignore[arg-type]
        embedder=CachingEmbedder(exploding, _Cache(env.store)),
        parser=ParserRegistry(),
        chunker=Chunker(env.settings),
    )
    with pytest.raises(EmbeddingUnavailableError):
        await service.ingest(DOC_V2_EXTRA.encode(), "handbook.md")

    # The embedder is called before any version is allocated, so this failure costs
    # nothing: no version number is spent and no row is written. That ordering is the
    # reason embedding happens outside the lock (D33).
    assert [row["state"] for row in env.store.versions("handbook")] == ["published"]
    assert env.store.get_document("handbook")["last_version"] == 1, "no version was burned"
    assert env.store.expected_vector_ids() == env.dense_ids()
    assert env.store.ingest_log("handbook")[-1]["error_code"] == "EMBEDDING_UNAVAILABLE"


# ---------------------------------------------------------------- test 8 (I3)


async def test_8_chroma_failure_during_staging_aborts_the_publish(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    before_eligible, before_lexical = env.eligible(), env.lexical_ids("installer")

    env.vectors.fail_on = "upsert"
    with pytest.raises(VectorStoreError):
        await env.service.ingest(DOC_V2_EXTRA.encode(), "handbook.md")

    assert env.eligible() == before_eligible
    assert env.lexical_ids("installer") == before_lexical
    assert env.lexical_ids("twenty megabytes") == []
    assert env.store.get_document("handbook")["current_version"] == 1
    assert env.store.expected_vector_ids() == env.dense_ids()


async def test_8b_a_chroma_upsert_that_silently_loses_vectors_aborts_the_publish(env: Any) -> None:
    """Rule 7 plus the VERIFY step: a partial write must not be published as complete."""
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    before = env.eligible()

    class SilentlyDropping(FlakyVectors):
        def upsert(self, items: Sequence[VectorItem]) -> None:
            self._inner.upsert(items[:1])  # writes one of many, reports success

    service = IngestionService(
        store=env.store,
        vectors=SilentlyDropping(env.real),  # type: ignore[arg-type]
        embedder=env.embedder,
        parser=ParserRegistry(),
        chunker=Chunker(env.settings),
    )
    with pytest.raises(VectorStoreError):
        await service.ingest(DOC_V2_EXTRA.encode(), "handbook.md")
    assert env.eligible() == before
    assert env.store.get_document("handbook")["current_version"] == 1


# ---------------------------------------------------------------- test 9


async def test_9_a_version_left_staging_is_failed_on_reconcile(env: Any) -> None:
    """Simulates a crash between stage and publish: the rows exist, nothing published."""
    first = await env.service.ingest(DOC_V1.encode(), "handbook.md")
    document = ParserRegistry().parse(DOC_V2_EXTRA.encode(), "handbook.md")
    chunks = Chunker(env.settings).chunk(document)
    version = env.store.begin_staging("handbook", "Handbook", "handbook.md", "md", "c", "p", "en")
    env.store.stage_version(
        "handbook", version, "c", document.text, chunks, [c.text for c in chunks]
    )

    assert env.store.version_state("handbook", version) == "staging"
    plan = await env.service.reconcile()

    assert env.store.version_state("handbook", version) == "failed"
    assert env.store.chunk_count("handbook", version) == 0
    # The staging version was purged before the plan was taken, so nothing is missing.
    assert plan.missing == () and plan.in_sync is True
    assert env.store.expected_vector_ids() == env.dense_ids(), "no strays left behind"
    assert env.eligible() == [f"handbook:v1:{i}" for i in range(first.chunks_added)]


async def test_9b_reconcile_removes_chroma_ids_sqlite_does_not_expect(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    env.real.upsert([VectorItem("ghost:v1:0", [1.0] * DIM, {"doc_id": "ghost"})])

    plan = await env.service.reconcile()
    # The plan reports what was FOUND, not what was repaired: a caller logging drift needs
    # the former, and a plan recomputed afterwards would always look clean.
    assert plan.orphaned == ("ghost:v1:0",)
    assert plan.in_sync is False
    assert "ghost:v1:0" not in env.dense_ids()
    assert (await env.service.reconcile()).in_sync is True


async def test_9c_reconcile_never_derives_sqlite_state_from_chroma(env: Any) -> None:
    """Rule 8: only SQLite decides. Chroma is repaired to match, never the reverse.

    A vector id for a document SQLite has never heard of must be deleted, and that
    document must stay absent. The reverse -- inventing a document because a vector
    exists -- is the failure mode this rules out.
    """
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    env.real.upsert(
        [
            VectorItem("phantom:v1:0", [1.0] * DIM, {"doc_id": "phantom", "doc_version": 1}),
            VectorItem("handbook:v99:0", [1.0] * DIM, {"doc_id": "handbook", "doc_version": 99}),
        ]
    )
    before = env.store.get_document("handbook")

    await env.service.reconcile()
    assert env.store.get_document("phantom") is None, "a document was invented from a vector"
    assert env.store.get_document("handbook") == before, "SQLite was rewritten from Chroma"
    assert env.store.get_document("handbook")["current_version"] == 1, "no version invented"
    assert not any(
        vector_id.startswith(("phantom:", "handbook:v99:")) for vector_id in env.dense_ids()
    )


# ---------------------------------------------------------------- test 10 (I10)


async def test_10_rebuild_restores_dense_search_with_zero_embed_calls(env: Any) -> None:
    from qasystem.ingestion.reconcile import rebuild

    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    expected = env.dense_ids()
    env.fake.reset()

    empty = ChromaStore(env.real.path.parent / "rebuilt", model_id="fake-embedder", dimension=DIM)
    assert rebuild(env.store, empty, "fake-embedder") == len(expected)
    assert sorted(empty.list_ids()) == expected
    assert env.fake.texts_embedded == 0, "rebuild called the embedding API"
    empty.close()


# ---------------------------------------------------------------- test 11


async def test_11_a_query_during_an_update_never_sees_mixed_versions(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    seen: list[tuple[int, ...]] = []

    async def reader() -> None:
        # No lock: the read path is not the writer's problem. Whatever it sees must be one
        # whole version, because staging rows are ineligible by construction.
        for _ in range(40):
            ids = env.eligible()
            versions = {int(i.split(":v")[1].split(":")[0]) for i in ids}
            seen.append(tuple(sorted(versions)))
            await asyncio.sleep(0)

    await asyncio.gather(
        reader(), env.service.ingest(DOC_V2_EXTRA.encode(), "handbook.md"), reader()
    )
    assert seen, "the reader never ran"
    assert all(len(versions) == 1 for versions in seen), seen


async def test_11b_concurrent_ingestions_of_the_same_document_serialise(env: Any) -> None:
    results = await asyncio.gather(
        env.service.ingest(DOC_V1.encode(), "handbook.md"),
        env.service.ingest(DOC_V2_EXTRA.encode(), "handbook.md"),
        env.service.ingest(DOC_V1.encode(), "handbook.md"),
    )
    assert all(r.status in {"created", "updated", "unchanged"} for r in results)
    versions = [r.version for r in results]
    assert len(set(versions)) == len(versions), f"two ingestions got the same version: {versions}"
    current = env.store.get_document("handbook")["current_version"]
    assert env.eligible() == [i for i in env.eligible() if f":v{current}:" in i]


# ---------------------------------------------------------------- test 12


async def test_12_duplicate_chunks_keep_multiplicity(env: Any) -> None:
    repeated = (
        "# Repeated\n\n## A\n\nSame paragraph.\n\n"
        "## B\n\nSame paragraph.\n\n## C\n\nSame paragraph.\n"
    )
    result = await env.service.ingest(repeated.encode(), "dupes.md")
    texts = [row["text"] for row in env.store.chunks_for_version("dupes", result.version)]
    assert len([t for t in texts if t.strip() == "Same paragraph."]) >= 2

    # Growing the duplicates must embed the new copies, not skip them.
    env.fake.reset()
    grown = repeated + "\n## D\n\nSame paragraph.\n"
    again = await env.service.ingest(grown.encode(), "dupes.md")
    assert again.chunks_added >= 1


async def test_12b_a_reorder_only_edit_costs_zero_embed_calls(env: Any) -> None:
    original = "# Guide\n\n## Install\n\nInstall the widget.\n\n## Configure\n\nSet the option.\n"
    reordered = "# Guide\n\n## Configure\n\nSet the option.\n\n## Install\n\nInstall the widget.\n"
    first = await env.service.ingest(original.encode(), "guide.md")
    assert first.chunks_added == 2
    env.fake.reset()

    again = await env.service.ingest(reordered.encode(), "guide.md")
    assert again.status == "updated"
    assert again.embed_requests == 0, "a reorder must not cost an embedding call"
    assert env.fake.texts_embedded == 0
    assert again.chunks_reused == 2
    assert env.lexical_ids("widget")


# ---------------------------------------------------------------- the ingest log


async def test_the_ingest_log_records_counts_and_no_text(env: Any) -> None:
    await env.service.ingest(DOC_V1.encode(), "handbook.md")
    await env.service.ingest(DOC_V1_EDITED.encode(), "handbook.md")
    await env.service.ingest(DOC_V1_EDITED.encode(), "handbook.md")  # same bytes as current

    rows = env.store.ingest_log("handbook")
    assert [row["action"] for row in rows] == ["publish", "publish", "unchanged"]
    assert rows[0]["chunks_added"] > 0 and rows[0]["chunks_reused"] == 0
    assert rows[1]["prev_version"] == 1 and rows[1]["new_version"] == 2
    assert rows[1]["chunks_added"] == 1
    assert rows[2]["embed_requests"] == 0
    assert rows[2]["new_version"] == 2, "an unchanged upload allocates no new version"
    assert all(row["duration_ms"] is not None for row in rows)
    joined = str(rows)
    for secret in ("installer", "ERR-404", "kernel version", "widget"):
        assert secret not in joined, f"the log leaked document text: {secret}"


async def test_a_failed_ingest_is_logged_with_its_error_code(env: Any) -> None:
    exploding = ExplodingEmbedder(FakeEmbedder(dimension=DIM), on_call=1)
    service = IngestionService(
        store=env.store,
        vectors=env.vectors,  # type: ignore[arg-type]
        embedder=CachingEmbedder(exploding, _Cache(env.store)),
        parser=ParserRegistry(),
        chunker=Chunker(env.settings),
    )
    with pytest.raises(EmbeddingUnavailableError):
        await service.ingest(DOC_V1.encode(), "handbook.md")
    row = env.store.ingest_log("handbook")[-1]
    assert row["status"] == "failed"
    assert row["error_code"] == "EMBEDDING_UNAVAILABLE"


# ---------------------------------------------------------------- shape


async def test_the_result_reports_what_happened(env: Any) -> None:
    created = await env.service.ingest(DOC_V1.encode(), "handbook.md")
    assert created.status == "created"
    assert created.doc_id == "handbook"
    assert created.duration_ms >= 0
    assert created.embed_requests == 1
    assert env.real.count() == created.chunks_added


async def test_an_explicit_doc_id_overrides_the_filename_slug(env: Any) -> None:
    result = await env.service.ingest(DOC_V1.encode(), "Hand Book!.md", doc_id="custom")
    assert result.doc_id == "custom"
    assert env.store.get_document("custom") is not None


def test_the_default_doc_id_is_a_filesystem_safe_slug() -> None:
    assert default_doc_id("Hand Book!.md") == "hand-book"
    assert default_doc_id("a/b/c/My_File.TXT") == "my-file"
    assert default_doc_id("...")
