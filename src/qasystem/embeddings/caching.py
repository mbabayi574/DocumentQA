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
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Protocol

from qasystem.domain.ports import Embedder

logger = logging.getLogger(__name__)

DEFAULT_QUERY_LRU_SIZE = 128


def input_hash(text: str) -> str:
    """sha256 of the exact embedded string; the cache key, with no normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache(Protocol):
    """Persistent vector store keyed by ``(input_hash, model_id)``; P5 implements it in SQLite."""

    def get_many(self, model_id: str, input_hashes: Sequence[str]) -> dict[str, list[float]]: ...

    def put_many(self, model_id: str, rows: Mapping[str, Sequence[float]]) -> None: ...


class CachingEmbedder:
    """Wraps an ``Embedder`` so repeated inputs are served from the cache."""

    def __init__(
        self,
        inner: Embedder,
        cache: EmbeddingCache,
        *,
        query_lru_size: int = DEFAULT_QUERY_LRU_SIZE,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._query_lru_size = query_lru_size
        self._query_lru: MutableMapping[str, list[float]] = {}

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimension(self) -> int:
        return self._inner.dimension

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
            self._put(self.model_id, dict(zip(missing, fresh, strict=True)))
            found.update(zip(missing, fresh, strict=True))
        return [found[digest] for digest in hashes]

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query, memoized in a bounded LRU so repeated questions are free."""
        digest = input_hash(text)
        hit = self._query_lru.get(digest)
        if hit is not None:
            return hit
        vector = (await self.embed([text]))[0]
        if len(self._query_lru) >= self._query_lru_size:
            # Cheap eviction: dicts keep insertion order, so the first key is the oldest.
            del self._query_lru[next(iter(self._query_lru))]
        self._query_lru[digest] = vector
        return vector

    async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._inner.embed(texts)

    def _get(self, model_id: str, hashes: Sequence[str]) -> dict[str, list[float]]:
        try:
            return self._cache.get_many(model_id, hashes)
        except Exception as exc:
            logger.warning(
                "embedding cache read failed (%s); embedding everything", type(exc).__name__
            )
            return {}

    def _put(self, model_id: str, rows: Mapping[str, Sequence[float]]) -> None:
        try:
            self._cache.put_many(model_id, rows)
        except Exception as exc:
            logger.warning("embedding cache write failed (%s); continuing", type(exc).__name__)
