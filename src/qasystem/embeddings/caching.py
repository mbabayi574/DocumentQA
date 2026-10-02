"""Caching embedder (plan.md P4, invariants I4/I5/I9).

One call in, one vector per item out, but the API is only called for inputs whose
``sha256`` is missing from the cache. That is what makes I5 true: re-ingesting a
document whose chunks did not change costs **zero** embedding requests, and I10's
``rebuild`` can restore dense search from these rows alone.

The cache is keyed by ``(input_hash, model_id)`` — the exact string that was embedded
plus the model that embedded it. Keying on the model is I9: two models must never read
each other's vectors, even at the same dimension.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from typing import Protocol

from qasystem.domain.ports import Embedder

logger = logging.getLogger(__name__)


def input_hash(text: str) -> str:
    """sha256 of the exact embedded string; the cache key, with no normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache(Protocol):
    """Persistent vector store keyed by ``(input_hash, model_id)``.

    Named after ``SqliteStore``'s own methods and shaped like them, so the store satisfies
    this structurally and no adapter class stands between them (D75). The in-memory fakes
    in the tests are the second implementation that earns the port its existence (§10r16).
    """

    def get_embeddings(
        self, model_id: str, input_hashes: Sequence[str]
    ) -> dict[str, list[float]]: ...

    def put_embeddings(
        self, model_id: str, rows: Iterable[tuple[str, Sequence[float]]]
    ) -> None: ...


class CachingEmbedder:
    """Wraps an ``Embedder`` so repeated inputs are served from the cache."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def requests(self) -> int:
        """Network requests issued, delegated to the wrapped embedder (D46).

        Deliberately *not* a count of this object's `embed()` calls. One such call fans out
        into up to `MAX_ITEMS_PER_BATCH` HTTP requests, so counting calls understated a real
        1225-chunk ingest as 5 requests instead of 39 -- a 32x error in the number
        `ingest_log` records and §9.2 reports. Caching is unaffected: the *delta* is what
        the caller wants, and a cache hit issues no request at all.
        """
        return self._inner.requests

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        hashes = [input_hash(text) for text in texts]
        # dict.fromkeys de-duplicates while keeping first-appearance order.
        unique = dict.fromkeys(hashes)
        found = self._get(self.model_id, list(unique))
        missing = [digest for digest in unique if digest not in found]
        if missing:
            by_hash = dict(zip(unique, dict.fromkeys(texts), strict=True))
            fresh = await self._embed([by_hash[digest] for digest in missing])
            self._put(self.model_id, list(zip(missing, fresh, strict=True)))
            found.update(zip(missing, fresh, strict=True))
        return [found[digest] for digest in hashes]

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._inner.embed(texts)

    def _get(self, model_id: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        try:
            return self._cache.get_embeddings(model_id, hashes)
        except Exception as exc:
            logger.warning(
                "embedding cache read failed (%s); embedding everything", type(exc).__name__
            )
            return {}

    def _put(self, model_id: str, rows: Iterable[tuple[str, Sequence[float]]]) -> None:
        try:
            self._cache.put_embeddings(model_id, rows)
        except Exception as exc:
            logger.warning("embedding cache write failed (%s); continuing", type(exc).__name__)
