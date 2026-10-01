"""P4: the HTTP embedding client (respx, no real network).

Every §2.1 behaviour the client depends on is pinned here, so a provider change fails
a test instead of silently corrupting a vector: the non-permutation ``index`` field,
three batching bounds, absent ``Retry-After``, 403-with-model-list, and the 8192-token
per-item context error.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
import respx

from qasystem.embeddings.client import EmbeddingClient, plan_batches
from qasystem.embeddings.rate_limit import RateLimiter
from qasystem.errors import ConfigError, EmbeddingAuthError, EmbeddingUnavailableError

BASE = "https://provider.test/v1"
SECRET = "sk-super-secret-token-value"
MODEL = "Bge-m3"
DIM = 4

REMOTE = {
    "embedding_base_url": BASE,
    "embedding_api_key": SECRET,
    "embedding_model": MODEL,
    "embedding_provider": "remote",
    "max_retries": 3,
    "rate_limit_per_min": 120,
}


# ---------------------------------------------------------------- helpers


def vector(value: float) -> list[float]:
    return [value, 0.0, 0.0, 0.0]


def ok_body(vectors: Sequence[Sequence[float]], *, model: str = MODEL) -> dict[str, Any]:
    """A well-formed Bge-style response: ``index`` present and a permutation."""
    return {
        "id": "emb-1",
        "object": "list",
        "created": 1,
        "model": model,
        "data": [
            {"object": "embedding", "index": i, "embedding": list(v)} for i, v in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


def route_models(ids: Sequence[str] = (MODEL,)) -> respx.Route:
    return respx.get(f"{BASE}/models").mock(
        return_value=httpx.Response(
            200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
        )
    )


def route_embed(vectors: Sequence[Sequence[float]], *, model: str = MODEL) -> respx.Route:
    return respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(200, json=ok_body(vectors, model=model))
    )


def route_counting_embed(start: float = 1.0) -> respx.Route:
    """Echo one vector per requested item, numbered so batches stay distinguishable."""
    counter = itertools.count(start)

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.content)["input"])
        return httpx.Response(200, json=ok_body([vector(next(counter)) for _ in range(count)]))

    return respx.post(f"{BASE}/embeddings").mock(side_effect=handler)


class CountingLimiter(RateLimiter):
    """A limiter that never waits, so tests can count how often it was consulted."""

    def __init__(self) -> None:
        super().__init__(10**9)
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1


def make_client(settings_factory: Any, **overrides: Any) -> EmbeddingClient:
    return EmbeddingClient(settings_factory(**{**REMOTE, **overrides}), limiter=RateLimiter(10**9))


def started(settings_factory: Any, **overrides: Any) -> EmbeddingClient:
    """A client that has already passed discovery, with a probed dimension."""
    instance = make_client(settings_factory, **overrides)
    instance.model_id = MODEL
    instance.dimension = DIM
    return instance


def embed_calls() -> list[list[str]]:
    return [
        json.loads(call.request.content)["input"]
        for call in respx.calls
        if "embeddings" in str(call.request.url)
    ]


# ---------------------------------------------------------------- discovery


@respx.mock
async def test_startup_discovery_accepts_the_exact_model_id(settings_factory: Any) -> None:
    route_models(["Bge-m3", "Gemini-embedding-001"])
    route_counting_embed()
    instance = await make_client(settings_factory).start()
    assert instance.model_id == "Bge-m3"
    assert instance.dimension == DIM
    assert instance.available_models == ("Bge-m3", "Gemini-embedding-001")


@respx.mock
async def test_an_unlisted_model_fails_fast_and_lists_the_available_ids(
    settings_factory: Any,
) -> None:
    route_models(["Embedding-3-Small", "Bge-m3"])
    with pytest.raises(EmbeddingAuthError) as exc:
        await make_client(settings_factory, embedding_model="bge-m3").start()
    assert "Bge-m3" in str(exc.value) and "Embedding-3-Small" in str(exc.value)
    assert SECRET not in str(exc.value)
    assert exc.value.code == "EMBEDDING_AUTH"


@respx.mock
async def test_a_malformed_models_response_is_a_typed_error(settings_factory: Any) -> None:
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json={"nope": True}))
    with pytest.raises(EmbeddingUnavailableError, match="models"):
        await make_client(settings_factory).start()


@respx.mock
async def test_a_configured_dimension_that_disagrees_with_the_probe_is_refused(
    settings_factory: Any,
) -> None:
    route_models()
    route_counting_embed()
    with pytest.raises(ConfigError, match="EMBEDDING_DIMENSION"):
        await make_client(settings_factory, embedding_dimension=1536).start()


@respx.mock
async def test_the_matching_configured_dimension_is_accepted(settings_factory: Any) -> None:
    route_models()
    route_counting_embed()
    instance = await make_client(settings_factory, embedding_dimension=DIM).start()
    assert instance.dimension == DIM


# ---------------------------------------------------------------- reassembly


@respx.mock
async def test_a_shuffled_index_is_reassembled_into_input_order(settings_factory: Any) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": MODEL,
                "data": [
                    {"index": 2, "embedding": vector(3.0)},
                    {"index": 0, "embedding": vector(1.0)},
                    {"index": 1, "embedding": vector(2.0)},
                ],
            },
        )
    )
    got = await started(settings_factory).embed(["a", "b", "c"])
    assert [v[0] for v in got] == [1.0, 2.0, 3.0]


@respx.mock
async def test_a_non_permutation_index_falls_back_to_response_order(
    settings_factory: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """§2.1: Gemini returns ``index: [0,0,0,0]``; trusting it would scramble every vector."""
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": MODEL,
                "data": [
                    {"index": 0, "embedding": vector(1.0)},
                    {"index": 0, "embedding": vector(2.0)},
                    {"index": 0, "embedding": vector(3.0)},
                ],
            },
        )
    )
    with caplog.at_level("WARNING", logger="qasystem.embeddings.client"):
        got = await started(settings_factory).embed(["a", "b", "c"])
    assert [v[0] for v in got] == [1.0, 2.0, 3.0]
    assert "index" in caplog.text


@respx.mock
async def test_a_missing_index_field_falls_back_to_response_order(settings_factory: Any) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"model": MODEL, "data": [{"embedding": vector(1.0)}, {"embedding": vector(2.0)}]},
        )
    )
    got = await started(settings_factory).embed(["a", "b"])
    assert [v[0] for v in got] == [1.0, 2.0]


@respx.mock
async def test_a_wrong_length_vector_is_rejected_rather_than_stored(settings_factory: Any) -> None:
    """A short vector would make cosine meaningless, so fail instead of padding."""
    route_embed([vector(1.0), [1.0, 2.0]])
    with pytest.raises(EmbeddingUnavailableError, match="dimension"):
        await started(settings_factory).embed(["a", "b"])


@respx.mock
async def test_a_short_response_is_rejected(settings_factory: Any) -> None:
    route_embed([vector(1.0)])
    with pytest.raises(EmbeddingUnavailableError, match="2"):
        await started(settings_factory).embed(["a", "b"])


# ---------------------------------------------------------------- batching


def test_batches_never_exceed_the_item_cap() -> None:
    batches = plan_batches(
        ["x" * 10] * 10, max_chars_per_request=10**6, max_items=3, max_chars_per_item=10**6
    )
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_batches_never_exceed_the_request_char_budget() -> None:
    batches = plan_batches(
        ["x" * 100] * 10, max_chars_per_request=250, max_items=99, max_chars_per_item=10**6
    )
    assert [sum(len(t) for t in b) for b in batches] == [200] * 5


def test_an_oversized_single_item_is_its_own_batch_not_a_silent_truncation() -> None:
    batches = plan_batches(
        ["x" * 500, "y"], max_chars_per_request=100, max_items=9, max_chars_per_item=10**6
    )
    assert batches[0] == ["x" * 500]
    assert batches[1] == ["y"]


def test_an_item_over_the_per_item_cap_is_rejected_before_any_request() -> None:
    """§2.1: Bge-m3 fails above ~40k chars with an 8192-token context error."""
    with pytest.raises(ValueError, match="MAX_CHARS_PER_ITEM"):
        plan_batches(["x" * 300], max_chars_per_request=10**6, max_items=9, max_chars_per_item=200)


def test_batching_covers_every_input_exactly_once() -> None:
    texts = [f"item-{i}" for i in range(37)]
    batches = plan_batches(texts, max_chars_per_request=200, max_items=4, max_chars_per_item=50)
    assert [t for batch in batches for t in batch] == texts


@respx.mock
async def test_embed_actually_batches_and_preserves_global_order(settings_factory: Any) -> None:
    route_counting_embed()
    got = await started(settings_factory, max_items_per_batch=2).embed(["a", "b", "c", "d"])
    assert [v[0] for v in got] == [1.0, 2.0, 3.0, 4.0]
    assert embed_calls() == [["a", "b"], ["c", "d"]]


@respx.mock
async def test_an_empty_input_makes_no_request(settings_factory: Any) -> None:
    route_embed([])
    assert await started(settings_factory).embed([]) == []
    assert respx.calls.call_count == 0


# ---------------------------------------------------------------- retries


@respx.mock
async def test_a_429_without_retry_after_is_retried_and_succeeds(settings_factory: Any) -> None:
    """§2.1 measured Retry-After absent, so backoff is the only path that works."""
    route = respx.post(f"{BASE}/embeddings")
    route.side_effect = [
        httpx.Response(429, json={"error": "rate limit"}),
        httpx.Response(200, json=ok_body([vector(1.0)])),
    ]
    got = await started(settings_factory).embed(["a"])
    assert got[0][0] == 1.0
    assert route.call_count == 2


@respx.mock
async def test_retry_after_seconds_is_honoured_when_present(settings_factory: Any) -> None:
    route = respx.post(f"{BASE}/embeddings")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "slow down"}),
        httpx.Response(200, json=ok_body([vector(1.0)])),
    ]
    got = await started(settings_factory).embed(["a"])
    assert got[0][0] == 1.0
    assert route.call_count == 2


@respx.mock
async def test_a_500_that_never_recovers_becomes_a_typed_unavailable_error(
    settings_factory: Any,
) -> None:
    respx.post(f"{BASE}/embeddings").mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(EmbeddingUnavailableError) as exc:
        await started(settings_factory, max_retries=2).embed(["a"])
    assert exc.value.code == "EMBEDDING_UNAVAILABLE"
    assert "500" in str(exc.value)


@respx.mock
async def test_a_timeout_is_retried_then_reported(settings_factory: Any) -> None:
    respx.post(f"{BASE}/embeddings").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(EmbeddingUnavailableError, match="timed out"):
        await started(settings_factory, max_retries=1).embed(["a"])


@respx.mock
async def test_a_401_fails_fast_without_retrying(settings_factory: Any) -> None:
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(401, json={"error": "missing or invalid token"})
    )
    with pytest.raises(EmbeddingAuthError) as exc:
        await started(settings_factory).embed(["a"])
    assert route.call_count == 1
    assert exc.value.code == "EMBEDDING_AUTH"
    assert SECRET not in str(exc.value)


@respx.mock
async def test_a_403_model_not_available_surfaces_the_allowed_ids(settings_factory: Any) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            403,
            json={"error": "model 'nope' is not available. Allowed: Bge-m3, Gemini-embedding-001"},
        )
    )
    with pytest.raises(EmbeddingAuthError) as exc:
        await started(settings_factory, max_retries=3).embed(["a"])
    assert "Bge-m3" in str(exc.value)
    assert exc.value.details.get("available_models") == ["Bge-m3", "Gemini-embedding-001"]


@respx.mock
async def test_a_400_context_overflow_is_not_retried(settings_factory: Any) -> None:
    """§2.1: retrying a too-long item just spends quota on the same error."""
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(400, json={"error": "maximum context length is 8192 token"})
    )
    with pytest.raises(EmbeddingUnavailableError, match="context length"):
        await started(settings_factory).embed(["x" * 100])
    assert route.call_count == 1


@respx.mock
async def test_an_empty_input_list_is_a_provider_400_not_a_bug(settings_factory: Any) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(400, json={"error": "please provide at least one prompt"})
    )
    with pytest.raises(EmbeddingUnavailableError, match="at least one prompt"):
        await started(settings_factory).embed(["a"])


# ---------------------------------------------------------------- misc


@respx.mock
async def test_every_request_carries_the_bearer_token(settings_factory: Any) -> None:
    route_embed([vector(1.0)])
    await started(settings_factory).embed(["a"])
    assert respx.calls.last.request.headers["Authorization"] == f"Bearer {SECRET}"


@respx.mock
async def test_the_token_never_appears_in_an_error_body(settings_factory: Any) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(500, text=f"key {SECRET} leaked")
    )
    with pytest.raises(EmbeddingUnavailableError) as exc:
        await started(settings_factory, max_retries=0).embed(["a"])
    assert SECRET not in str(exc.value)
    assert SECRET not in json.dumps(exc.value.details)


@respx.mock
async def test_the_rate_limiter_is_consulted_once_per_request(settings_factory: Any) -> None:
    route_counting_embed()
    gate = CountingLimiter()
    instance = started(settings_factory, max_items_per_batch=2)
    instance.limiter = gate
    await instance.embed(["a", "b", "c"])
    assert gate.acquired == 2  # one per HTTP request, not one per item
