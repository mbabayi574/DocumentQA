"""P0: FakeEmbedder is deterministic, L2-normalised and offline."""

from __future__ import annotations

import math

import pytest

from qasystem.domain.ports import Embedder
from qasystem.embeddings.fake import FakeEmbedder

TEXT = "The retrieval system uses a single SQLite snapshot per query."


def test_satisfies_the_embedder_port() -> None:
    embedder: Embedder = FakeEmbedder(dimension=16)
    assert embedder.model_id == "fake-embedder"
    assert embedder.dimension == 16


async def test_is_deterministic_across_instances() -> None:
    first = FakeEmbedder(dimension=16)
    second = FakeEmbedder(dimension=16)
    assert await first.embed([TEXT, "another"]) == await second.embed([TEXT, "another"])


async def test_is_deterministic_across_repeated_calls() -> None:
    embedder = FakeEmbedder(dimension=16)
    assert await embedder.embed([TEXT]) == await embedder.embed([TEXT])


async def test_vectors_are_l2_normalised_and_sized() -> None:
    vector = (await FakeEmbedder(dimension=32).embed([TEXT]))[0]
    assert len(vector) == 32
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


async def test_empty_text_still_yields_a_unit_vector() -> None:
    vector = (await FakeEmbedder(dimension=8).embed([""]))[0]
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


async def test_order_is_preserved() -> None:
    embedder = FakeEmbedder(dimension=16)
    texts = ["alpha", "beta", "gamma"]
    embedded = await embedder.embed(texts)
    assert embedded == [(await embedder.embed([text]))[0] for text in texts]


async def test_texts_sharing_tokens_are_more_similar_than_unrelated_texts() -> None:
    embedder = FakeEmbedder(dimension=64)
    query = (await embedder.embed(["sqlite snapshot retrieval"]))[0]
    close = (await embedder.embed(["the retrieval reads one sqlite snapshot"]))[0]
    far = (await embedder.embed(["Persian: سیستم بازیابی معنایی متن"]))[0]
    assert _dot(query, close) > _dot(query, far)


async def test_counts_requests_and_texts() -> None:
    embedder = FakeEmbedder(dimension=8)
    await embedder.embed(["a", "b", "c"])
    assert (embedder.requests, embedder.texts_embedded) == (1, 3)
    embedder.reset()
    assert (embedder.requests, embedder.texts_embedded) == (0, 0)


def test_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FakeEmbedder(dimension=0)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
