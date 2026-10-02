"""P4: CachingEmbedder — I5 (unchanged chunks cost no requests) and I4 (re-upload is free)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from qasystem.embeddings.caching import CachingEmbedder
from qasystem.embeddings.fake import FakeEmbedder


class CountingCache:
    """In-memory stand-in for the SQLite ``embedding_cache`` table (P5 owns the real one)."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], list[float]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get_embeddings(self, model_id: str, input_hashes: Sequence[str]) -> dict[str, list[float]]:
        found = {}
        for digest in input_hashes:
            row = self.rows.get((model_id, digest))
            if row is None:
                self.misses += 1
            else:
                self.hits += 1
                found[digest] = row
        return found

    def put_embeddings(self, model_id: str, rows: Iterable[tuple[str, Sequence[float]]]) -> None:
        self.writes += 1
        for digest, vector in rows:
            self.rows[(model_id, digest)] = vector


@pytest.fixture
def inner() -> FakeEmbedder:
    return FakeEmbedder(dimension=8)


@pytest.mark.invariant  # I5
async def test_a_miss_embeds_and_writes_back(inner: FakeEmbedder) -> None:
    cache = CountingCache()
    embedder = CachingEmbedder(inner, cache)
    got = await embedder.embed(["alpha", "beta"])
    assert len(got) == 2
    assert inner.texts_embedded == 2
    assert cache.writes == 1


async def test_a_cache_hit_causes_zero_requests(inner: FakeEmbedder) -> None:
    """I5: a re-ingest of identical text must not touch the embedding API."""
    cache = CountingCache()
    embedder = CachingEmbedder(inner, cache)
    await embedder.embed(["alpha", "beta"])
    inner.reset()

    again = await embedder.embed(["alpha", "beta"])
    assert inner.requests == 0
    assert again == [inner._vector("alpha"), inner._vector("beta")]


async def test_duplicates_in_one_call_are_embedded_once(inner: FakeEmbedder) -> None:
    """Two identical chunks must not be sent twice in the same request."""
    embedder = CachingEmbedder(inner, CountingCache())
    got = await embedder.embed(["alpha", "alpha", "alpha"])
    assert inner.texts_embedded == 1
    assert got[0] == got[1] == got[2]


async def test_a_partial_hit_embeds_only_the_misses(inner: FakeEmbedder) -> None:
    cache = CountingCache()
    await CachingEmbedder(inner, cache).embed(["alpha"])
    inner.reset()
    got = await CachingEmbedder(inner, cache).embed(["alpha", "beta", "alpha"])
    assert inner.texts_embedded == 1
    assert len(got) == 3


async def test_a_different_model_id_never_reuses_another_models_vectors() -> None:
    """I9: the cache key includes the model, so dimensions cannot mix."""
    cache = CountingCache()
    first = FakeEmbedder(dimension=8, model_id="Bge-m3")
    second = FakeEmbedder(dimension=4, model_id="other-model")
    await CachingEmbedder(first, cache).embed(["alpha"])
    second.reset()
    got = await CachingEmbedder(second, cache).embed(["alpha"])
    assert second.texts_embedded == 1
    assert len(got[0]) == 4


async def test_an_empty_call_is_a_no_op(inner: FakeEmbedder) -> None:
    embedder = CachingEmbedder(inner, CountingCache())
    assert await embedder.embed([]) == []
    assert inner.requests == 0


async def test_the_embedder_exposes_the_inners_model_and_dimension() -> None:
    inner = FakeEmbedder(dimension=8, model_id="Bge-m3")
    embedder = CachingEmbedder(inner, CountingCache())
    assert embedder.model_id == "Bge-m3"
    assert embedder.dimension == 8


async def test_a_cache_repository_failure_does_not_lose_the_vectors() -> None:
    """The cache is an optimisation: a write fault must not fail the ingest."""

    class BrokenCache(CountingCache):
        def put_embeddings(
            self, model_id: str, rows: Iterable[tuple[str, Sequence[float]]]
        ) -> None:
            raise RuntimeError("disk full")

    inner = FakeEmbedder(dimension=8)
    got = await CachingEmbedder(inner, BrokenCache()).embed(["alpha"])
    assert len(got[0]) == 8


async def test_cache_keys_are_the_sha256_of_the_exact_embedded_string() -> None:
    import hashlib

    cache = CountingCache()
    inner = FakeEmbedder(dimension=8)
    await CachingEmbedder(inner, cache).embed(["alpha"])
    digest = hashlib.sha256(b"alpha").hexdigest()
    assert (inner.model_id, digest) in cache.rows


async def test_a_whitespace_change_is_a_different_cache_entry() -> None:
    """The key is the exact string, so a breadcrumb change correctly invalidates."""
    cache = CountingCache()
    inner = FakeEmbedder(dimension=8)
    embedder = CachingEmbedder(inner, cache)
    await embedder.embed(["Handbook > Install"])
    await embedder.embed(["Handbook >Install"])
    assert inner.texts_embedded == 2


# ---------------------------------------------------------------- D46


async def test_requests_counts_network_requests_not_embed_calls() -> None:
    """One `embed()` call fans out into batches; the counter must count the batches.

    Counting calls understated a real 1225-chunk ingest as 5 instead of 39 -- a 32x error in
    the number `ingest_log` records and §9.2 reports (D46).
    """

    class FanOut:
        """One `embed()` call, one request per item -- what MAX_ITEMS_PER_BATCH does live."""

        model_id = "fanout"
        dimension = 4

        def __init__(self) -> None:
            self.requests = 0

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.requests += len(texts)
            return [[0.0] * self.dimension for _ in texts]

    inner = FanOut()
    embedder = CachingEmbedder(inner, CountingCache())
    before = embedder.requests

    await embedder.embed(["one", "two", "three"])

    assert embedder.requests - before == 3, "not one: the fan-out is invisible"
    # And a cache hit still costs nothing, which is what the delta in ingest relies on.
    again = embedder.requests
    await embedder.embed(["one", "two", "three"])
    assert embedder.requests == again
