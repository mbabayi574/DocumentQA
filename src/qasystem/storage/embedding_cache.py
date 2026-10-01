"""The persistent embedding cache, on SQLite (plan.md P4's ``EmbeddingCache`` port).

Ten lines and no state of its own: ``SqliteStore`` already stores and reads
``embedding_cache`` rows, so the only thing missing was the two-method shape the port
asks for. Ingest and ``rebuild`` both run through this, which is what makes I5 (an
unchanged chunk costs zero requests) and I10 (``rebuild`` needs no API) true of the real
wiring rather than only of a test double.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from qasystem.storage.sqlite_store import SqliteStore


class SqliteEmbeddingCache:
    """``EmbeddingCache`` backed by the ``embedding_cache`` table."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def get_many(self, model_id: str, input_hashes: Sequence[str]) -> dict[str, list[float]]:
        return self._store.get_embeddings(model_id, input_hashes)

    def put_many(self, model_id: str, rows: Mapping[str, Sequence[float]]) -> None:
        self._store.put_embeddings(model_id, rows.items())
