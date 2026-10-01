"""Reconcile primitives and I10's ``rebuild`` (plan.md P5, P6 rule 8).

``plan_reconcile`` compares what SQLite says should exist with what the vector store
holds, and **changes nothing**. That asymmetry is deliberate: the vector store is a
derived index, so it may be wrong in either direction, and only SQLite is allowed to
decide the answer. P6 decides what to do with a plan; this module only produces one.

``rebuild`` is the proof that the index is disposable. It reads vectors out of
``embedding_cache`` and re-upserts them, so restoring dense search after deleting
``data/chroma/`` costs **zero** embedding API calls.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qasystem.domain.models import VectorItem
from qasystem.storage.chroma_store import PAGE_SIZE, ChromaStore
from qasystem.storage.sqlite_store import SqliteStore


class EmbedderRef(Protocol):
    """Accepted by ``rebuild`` purely so a test can pass a tripwire and prove the
    zero-API-call guarantee (I10) instead of taking it on trust."""

    model_id: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class ReconcilePlan:
    """What differs between SQLite's expectation and the vector store's contents."""

    missing: tuple[str, ...]  # SQLite has a chunk row, Chroma has no vector
    orphaned: tuple[str, ...]  # Chroma has a vector SQLite does not expect
    total_expected: int

    @property
    def in_sync(self) -> bool:
        return not self.missing and not self.orphaned


def plan_reconcile(store: SqliteStore, vectors: ChromaStore) -> ReconcilePlan:
    """Read-only comparison. Never mutates either side."""
    expected = set(store.expected_vector_ids())
    actual = set(vectors.list_ids())
    return ReconcilePlan(
        missing=tuple(sorted(expected - actual)),
        orphaned=tuple(sorted(actual - expected)),
        total_expected=len(expected),
    )


def rebuild(
    store: SqliteStore,
    vectors: ChromaStore,
    model_id: str,
    *,
    embedder: EmbedderRef | None = None,
) -> int:
    """Re-upsert every chunk's vector from ``embedding_cache``. Returns rows restored.

    ``embedder`` is accepted and never called; that is the point of passing it. A chunk
    whose vector is not cached is skipped, and ``plan_reconcile`` then reports it as
    missing so the caller can decide to embed it — a rebuild reports honestly rather
    than half-restoring the index.
    """
    vectors.ensure_collection()
    restored = 0
    for batch in _batched(store.chunks_need_vectors(), PAGE_SIZE):
        cached = store.get_embeddings(model_id, [row["embed_input_hash"] for row in batch])
        items = [
            VectorItem(
                id=row["chunk_id"],
                vector=cached[row["embed_input_hash"]],
                metadata={
                    "doc_id": row["doc_id"],
                    "doc_version": row["doc_version"],
                    "ordinal": row["ordinal"],
                },
            )
            for row in batch
            if row["embed_input_hash"] in cached
        ]
        if items:
            vectors.upsert(items)
            restored += len(items)
    return restored


def _batched(rows: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]
