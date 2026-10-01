"""Local persistent Chroma adapter (plan.md §4.2 L1-L10, P5).

Chroma is a **derived, disposable** index. It holds vectors and three diagnostic
fields and never document text, because SQLite decides eligibility and quotes the
source. Deleting ``data/chroma/`` and rebuilding from ``embedding_cache`` restores
dense search with zero API calls (I10).

Cosine is configured on the collection and ``similarity = 1 - distance``, verified
against known vectors rather than assumed: identical → 0.0, orthogonal → 1.0 (§2.2).

Every method is blocking, as the port is. Callers off the event loop wrap it in
``asyncio.to_thread`` (L3).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from qasystem.domain.models import VectorHit, VectorItem
from qasystem.errors import VectorStoreError

MODEL_ID_KEY = "model_id"
DIMENSION_KEY = "dimension"
SPACE_KEY = "hnsw:space"
PAGE_SIZE = 500
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_LEGAL = re.compile(r"[a-z0-9_-]{3,63}")


def collection_name(model_id: str, dimension: int) -> str:
    """``chunks__<slug>__d<dimension>``: one collection per (model, dimension) (I9, L4).

    Two models at the same dimension must not share vectors, so the model is in the
    name, not just in the metadata.
    """
    slug = _SLUG_RE.sub("_", model_id.lower()).strip("_")
    name = f"chunks__{slug}__d{dimension}"
    return name[:63] if _LEGAL.fullmatch(name) else name[:63].rstrip("_-") or "chunks"


class ChromaStore:
    """``VectorStore`` over ``chromadb.PersistentClient``."""

    def __init__(self, path: str | Path, *, model_id: str, dimension: int) -> None:
        self.path = Path(path)
        self.model_id = model_id
        self.dimension = dimension
        self._collection_name = collection_name(model_id, dimension)
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------- lifecycle

    def _open(self) -> Any:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        try:
            return chromadb.PersistentClient(
                path=str(self.path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise VectorStoreError(
                f"cannot open the local vector store at {self.path}: {type(exc).__name__}"
            ) from None

    def _collection_for_write(self) -> Any:
        if self._collection is not None:
            return self._collection
        self._client = self._open()
        try:
            self._collection = self._client.get_or_create_collection(
                self._collection_name,
                metadata={
                    SPACE_KEY: "cosine",
                    MODEL_ID_KEY: self.model_id,
                    DIMENSION_KEY: self.dimension,
                },
            )
        except Exception as exc:
            raise VectorStoreError(f"cannot open collection: {type(exc).__name__}") from None
        self._assert_isolated()
        return self._collection

    def _assert_isolated(self) -> None:
        """I9: refuse to touch a collection built by a different model or dimension.

        A silent reuse would put 3072-d vectors into a 1024-d space, and the symptom
        would be quietly bad retrieval rather than an error.
        """
        stored = self._collection.metadata or {}
        model = stored.get(MODEL_ID_KEY)
        dimension = stored.get(DIMENSION_KEY)
        if model is not None and str(model) != self.model_id:
            raise VectorStoreError(
                f"collection {self._collection_name} was built for model {model!r}, "
                f"not {self.model_id!r}; refusing to mix models (I9)"
            )
        if dimension is not None and int(dimension) != self.dimension:
            raise VectorStoreError(
                f"collection {self._collection_name} has dimension {dimension}, "
                f"not {self.dimension}; refusing to mix dimensions (I9)"
            )

    @property
    def collection_name(self) -> str:
        """Exposed so a caller can log which index it is talking to."""
        return self._collection_name

    def ensure_collection(self) -> None:
        """Open (creating if needed) the collection for this model and dimension (L4)."""
        self._collection_for_write()

    def close(self) -> None:
        """Close the connection. Further use raises, which is the point."""
        self._collection = None
        if self._client is not None:
            # Chroma's Rust core holds the sqlite handle; dropping the reference is the
            # documented way to release it, and the restart test proves it works.
            self._client = None

    def heartbeat(self) -> int:
        """Chroma's liveness timestamp; raises VectorStoreError when unreachable (L9)."""
        client = self._client if self._client is not None else self._open()
        try:
            return int(client.heartbeat())
        except Exception as exc:
            raise VectorStoreError(f"vector store is unresponsive: {type(exc).__name__}") from None

    def ping(self) -> None:
        self.heartbeat()

    def count(self) -> int:
        """Vectors currently in the collection; 0 before anything is written."""
        try:
            return int(self._collection_for_write().count())
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"cannot count vectors: {type(exc).__name__}") from None

    # ------------------------------------------------------------- writes

    def upsert(self, items: Sequence[VectorItem]) -> None:
        """Insert or replace by id. Local and free, so reuse costs no API call."""
        if not items:
            return
        for entry in items:
            if len(entry.vector) != self.dimension:
                raise VectorStoreError(
                    f"vector {entry.id} has dimension {len(entry.vector)}, "
                    f"expected {self.dimension}"
                )
        try:
            self._collection_for_write().upsert(
                ids=[entry.id for entry in items],
                embeddings=[list(entry.vector) for entry in items],
                metadatas=[dict(entry.metadata) for entry in items],
            )
        except Exception as exc:
            raise VectorStoreError(f"upsert failed: {type(exc).__name__}") from None

    def delete_ids(self, ids: Sequence[str]) -> None:
        """Best-effort removal of specific vectors; unknown ids are ignored."""
        if not ids:
            return
        try:
            self._collection_for_write().delete(ids=list(ids))
        except Exception as exc:
            raise VectorStoreError(f"delete failed: {type(exc).__name__}") from None

    # ------------------------------------------------------------- reads

    def query(self, vector: Sequence[float], n: int) -> list[VectorHit]:
        """The ``n`` nearest vectors, clamped to the collection size (L7)."""
        if len(vector) != self.dimension:
            raise VectorStoreError(
                f"query vector has dimension {len(vector)}, expected {self.dimension}"
            )
        collection = self._collection_for_write()
        total = int(collection.count())
        if total == 0 or n <= 0:
            return []  # L7: an empty collection returns empty lists; do not query it
        # L7: clamp explicitly. Chroma clamps too, but then `n` no longer means anything.
        take = min(n, total)
        try:
            result = collection.query(query_embeddings=[list(vector)], n_results=take)
        except Exception as exc:
            raise VectorStoreError(f"query failed: {type(exc).__name__}") from None
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        # L5: cosine distance, so similarity = 1 - distance (§2.2 measured this).
        return [
            VectorHit(id=str(vector_id), similarity=1.0 - float(distance))
            for vector_id, distance in zip(ids, distances, strict=False)
        ]

    def get_existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Which of ``ids`` are present. P6 verifies staging with this."""
        if not ids:
            return set()
        found: set[str] = set()
        collection = self._collection_for_write()
        try:
            for start in range(0, len(ids), PAGE_SIZE):
                page = ids[start : start + PAGE_SIZE]
                found.update(str(x) for x in (collection.get(ids=list(page)).get("ids") or []))
        except Exception as exc:
            raise VectorStoreError(f"get failed: {type(exc).__name__}") from None
        return found

    def list_ids(self) -> Iterator[str]:
        """Every id in the collection, paged (L7) so a large index never loads at once."""
        collection = self._collection_for_write()
        offset = 0
        while True:
            try:
                page = collection.get(limit=PAGE_SIZE, offset=offset).get("ids") or []
            except Exception as exc:
                raise VectorStoreError(f"list_ids failed: {type(exc).__name__}") from None
            if not page:
                return
            for vector_id in page:
                yield str(vector_id)
            offset += len(page)
