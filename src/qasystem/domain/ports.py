"""Ports (plan.md §4): the seams that let lanes A-D run in parallel.

A Protocol is added only when a second implementation or a test seam needs it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from qasystem.domain.models import ParsedDocument, VectorHit, VectorItem


class Embedder(Protocol):
    """Turns text into vectors. Implementations may be remote or offline."""

    model_id: str
    dimension: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input, in the same order."""
        ...


class VectorStore(Protocol):
    """Derived, rebuildable dense index (local persistent Chroma in production)."""

    def ensure_collection(self, model_id: str, dimension: int) -> None: ...

    def upsert(self, items: Sequence[VectorItem]) -> None: ...

    def query(self, vector: Sequence[float], n: int) -> list[VectorHit]: ...

    def get_existing_ids(self, ids: Sequence[str]) -> set[str]: ...

    def list_ids(self) -> Iterator[str]: ...

    def delete_ids(self, ids: Sequence[str]) -> None: ...

    def count(self) -> int: ...

    def ping(self) -> None:
        """Raise ``VectorStoreError`` when the store is unusable."""
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Bytes in, ``ParsedDocument`` out; one implementation per supported format."""

    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...
