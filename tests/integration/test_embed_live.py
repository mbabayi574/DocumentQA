"""P4: live provider probe. Skipped unless RUN_LIVE=1 and a real token is configured.

This is the only file allowed to touch the network (plan.md §0.6). It exists to confirm
the §2.1 measurements still hold; a failure here means the provider changed and the
pinned assumptions in the client are stale.

It never prints the token, and it asserts the token is absent from every error it
produces, so a live run cannot leak the secret into CI output.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest
from pydantic import SecretStr

from qasystem.config import load_settings
from qasystem.embeddings.client import EmbeddingClient
from qasystem.errors import ConfigError, EmbeddingAuthError

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1", reason="set RUN_LIVE=1 to call the real provider"
)


@pytest.fixture
async def live_client() -> Any:
    # A clean clone has no .env and therefore no token. That must skip, not error, so
    # `make live` is usable there (D31).
    try:
        settings = load_settings()
    except ConfigError as exc:
        pytest.skip(f"live run needs a configured provider: {exc.code}")
    if settings.is_fake_provider:
        pytest.skip("EMBEDDING_PROVIDER=fake; nothing live to call")
    if settings.embedding_api_key is None:
        pytest.skip("EMBEDDING_API_KEY is not set")
    async with EmbeddingClient(settings) as client:
        yield client


async def test_discovery_reports_the_configured_model_and_a_stable_dimension(
    live_client: EmbeddingClient,
) -> None:
    assert live_client.settings.embedding_model in live_client.available_models
    assert live_client.dimension > 0
    print(f"\nlive: {live_client.available_models} dimension={live_client.dimension}")


async def test_vectors_arrive_l2_normalized(live_client: EmbeddingClient) -> None:
    """plan.md §9.3: the server normalizes, so ingest skips normalizing (I9's premise)."""
    (vector,) = await live_client.embed(["a short English sentence"])
    norm = math.sqrt(sum(value * value for value in vector))
    assert abs(norm - 1.0) < 1e-3


async def test_batch_order_is_preserved(live_client: EmbeddingClient) -> None:
    texts = ["alpha text", "beta text", "gamma text", "delta text"]
    vectors = await live_client.embed(texts)
    assert len(vectors) == len(texts)
    assert len({tuple(v) for v in vectors}) == len(texts)  # four distinct vectors


async def test_an_item_at_the_configured_per_item_cap_is_accepted(
    live_client: EmbeddingClient,
) -> None:
    """MAX_CHARS_PER_ITEM=20000 must stay inside the measured 40 000-char ceiling."""
    item = "word " * (live_client.settings.max_chars_per_item // 5)
    vectors = await live_client.embed([item])
    assert len(vectors[0]) == live_client.dimension


async def test_a_far_oversized_item_is_refused_before_any_request(
    live_client: EmbeddingClient,
) -> None:
    """The per-item cap fires locally, so the provider is never asked to fail (I6)."""
    with pytest.raises(ValueError, match="MAX_CHARS_PER_ITEM"):
        await live_client.embed(["word " * 20_000])  # 100_000 chars


async def test_a_rejected_token_never_appears_in_the_error(live_client: EmbeddingClient) -> None:
    """A real 401 body is the one error path a bad key can actually reach."""
    bad = live_client.settings.model_copy(
        update={"embedding_api_key": SecretStr("sk-deliberately-wrong-token")}
    )
    with pytest.raises(EmbeddingAuthError) as exc:
        await EmbeddingClient(bad).start()
    assert "sk-deliberately-wrong-token" not in str(exc.value)
    assert exc.value.code == "EMBEDDING_AUTH"
