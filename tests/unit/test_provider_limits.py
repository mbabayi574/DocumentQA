"""Probe tests for the §2/§3 provider limits the shipped configuration depends on.

plan.md P10 asks for "a probe test for every §2/§3 behaviour production code depends on". The
audit found almost all of them already present — the ``index`` permutation fallback, 401 fail-fast,
403 with the allowed model list, 429 with and without ``Retry-After``, the dimension probe, the
token bucket, L2-normalized vectors, Chroma's ``n`` clamping, its empty-collection behaviour, its
metadata surviving a restart, and ``/ready`` failing on a broken dependency all have one.

The gap was the narrowest and the most dangerous: **nothing tied the shipped batching values to the
measured limits they were chosen from.** Every other limit is validated *relative to a constant in
``config.py``* — which is self-consistent and proves nothing about the provider. Raise
``MAX_CHARS_PER_ITEM`` to 50 000 and every validation still passes, while a 45 000-character chunk
that the provider rejects (§2 measured the ceiling at 40 949) starts failing ingest at runtime with
a 400 that only shows up on long documents.

These tests are the ones that would fail if the provider changed *or* if someone "optimised" a
number away from its measurement. They assert the **measured values**, not just the relationships.
"""

from __future__ import annotations

from typing import Any

import pytest

from qasystem.config import PROVIDER_MAX_CHARS_PER_REQUEST, PROVIDER_MAX_REQUESTS_PER_MIN, Settings

#: §2's measured table, transcribed. These are the numbers the shipped configuration was chosen
#: from; the tests below exist so that editing a default without re-measuring is a test failure
#: rather than a production 400.
MEASURED_ITEM_CEILING = 40_949  # binary search, 15 probes
MEASURED_REQUESTS_PER_MIN = 120  # jobTask.md C5
MEASURED_PROVIDER_CHARS_PER_REQUEST = 200_000  # §2, the request-level cap
MEASURED_THROUGHPUT_PEAK_ITEMS = 32  # 29.7 chunks/s, vs 23.6 at 64 and 28.2 at 128


def _remote() -> dict[str, Any]:
    return {
        "app_env": "prod",
        "embedding_provider": "remote",
        "embedding_api_key": "sk-probe-not-a-real-token",
        "embedding_model": "Bge-m3",
    }


# ------------------------------------------------------------------ the shipped values


def test_the_shipped_per_item_cap_sits_below_the_measured_ceiling(settings_factory: Any) -> None:
    """`MAX_CHARS_PER_ITEM` must stay under the provider's single-item ceiling.

    The existing tests only check the cap is *below* `MAX_CHARS_PER_REQUEST`, which is a
    self-consistency check. This one is against the measurement, and it carries the margin
    explicitly: 20 000 against 40 949 is 2.05x, so ordinary text variation cannot reach the
    boundary.
    """
    settings: Settings = settings_factory(**_remote())
    assert settings.max_chars_per_item < MEASURED_ITEM_CEILING
    margin = MEASURED_ITEM_CEILING / settings.max_chars_per_item
    assert margin >= 2.0, (
        f"the per-item cap is {settings.max_chars_per_item} against a measured ceiling of "
        f"{MEASURED_ITEM_CEILING}, a margin of {margin:.2f}x; §2 chose 2.05x and anything under "
        "2x means one long paragraph away from a runtime 400"
    )


def test_the_shipped_request_cap_leaves_room_under_the_providers(settings_factory: Any) -> None:
    """`MAX_CHARS_PER_REQUEST` must stay below the provider's own request cap.

    If ours ever exceeds theirs, the provider rejects a request we believed was legal, and the
    error is a 400 from a third party rather than something our own splitter could have caught.
    """
    settings: Settings = settings_factory(**_remote())
    assert settings.max_chars_per_request < MEASURED_PROVIDER_CHARS_PER_REQUEST
    assert PROVIDER_MAX_CHARS_PER_REQUEST == MEASURED_PROVIDER_CHARS_PER_REQUEST, (
        "the constant the config validator compares against no longer matches the measured "
        "provider cap, so the validation below it is validating against a fiction"
    )


def test_the_shipped_rate_limit_leaves_headroom_under_the_providers(settings_factory: Any) -> None:
    """§2 measured 120 req/min and chose 100 so a full re-ingest cannot become a 429 storm.

    The re-ingest figure matters as much as the limit: §2 measured the fixture corpus at 31
    req/min, so 100 is three times the real rate and a request that retries still has room.
    """
    settings: Settings = settings_factory(**_remote())
    assert settings.rate_limit_per_min < MEASURED_REQUESTS_PER_MIN
    assert PROVIDER_MAX_REQUESTS_PER_MIN == MEASURED_REQUESTS_PER_MIN
    assert settings.rate_limit_per_min >= 3 * 31, (
        "§2 measured a real re-ingest at 31 req/min; a limiter below 3x that would throttle a "
        "rebuild into the provider's own rate limit"
    )


def test_the_batch_size_is_still_the_measured_throughput_peak(settings_factory: Any) -> None:
    """§2 measured 32 items/request as the peak: 29.7 chunks/s, against 23.6 at 64 and 28.2 at
    128. It is tempting to "optimise" this upward, and the measurement says it is slower.

    Not a performance assertion — nothing here can be fast in a unit test. It is a guard on the
    *reason* the number is 32, so the next reader does not raise it on a hunch.
    """
    settings: Settings = settings_factory(**_remote())
    assert settings.max_items_per_batch == MEASURED_THROUGHPUT_PEAK_ITEMS


# ------------------------------------------------------------------ the splitter's reach


def test_a_batch_never_exceeds_the_configured_request_cap(settings_factory: Any) -> None:
    """The real invariant behind the three bounds: no request the client builds can exceed the cap.

    Checked by constructing the worst case rather than by trusting the arithmetic in
    ``plan_batches`` — a planner bug that put two oversized items in one request would still
    produce plausible-looking batches in a test that only checked counts.
    """
    from qasystem.embeddings.client import plan_batches

    settings: Settings = settings_factory(**_remote())
    worst = ["x" * settings.max_chars_per_item] * (settings.max_items_per_batch * 3)
    for batch in plan_batches(
        worst,
        max_chars_per_request=settings.max_chars_per_request,
        max_items=settings.max_items_per_batch,
        max_chars_per_item=settings.max_chars_per_item,
    ):
        assert sum(len(item) for item in batch) <= settings.max_chars_per_request
        assert len(batch) <= settings.max_items_per_batch
        assert all(len(item) <= settings.max_chars_per_item for item in batch)


def test_a_single_oversized_item_is_rejected_before_any_request_is_planned(
    settings_factory: Any,
) -> None:
    """D22: the per-item cap is enforced *before* any request, so an oversized chunk fails
    locally rather than burning quota on a predictable 400.

    Asserted on the plan rather than on a live call, because the property is that no request is
    built at all.

    Two things recorded rather than assumed. The exception is a bare ``ValueError``, not a typed
    domain error, and §5.3 would prefer a typed one -- but the path is unreachable from the API:
    the chunker caps a chunk at ``CHUNK_HARD_MAX_TOKENS x 1.5`` = 1 050 characters against a
    20 000-character guard, a 19x gap, so no ingested document can get here. Adding an error class
    for an input nothing can produce is the speculative work §10 rule 16 forbids, and the message
    names the cause and the remedy, so a future caller that can reach it has what it needs.
    The test pins the message because that is the part a reachable caller would rely on.
    """
    from qasystem.embeddings.client import plan_batches

    settings: Settings = settings_factory(**_remote())
    with pytest.raises(ValueError, match="MAX_CHARS_PER_ITEM"):
        plan_batches(
            ["x" * (settings.max_chars_per_item + 1)],
            max_chars_per_request=settings.max_chars_per_request,
            max_items=settings.max_items_per_batch,
            max_chars_per_item=settings.max_chars_per_item,
        )

    # And the gap that makes it unreachable is asserted, so a future chunker change that
    # narrowed it would be caught here rather than in production.
    assert settings.chunk_hard_max_tokens * 1.5 * 4 < settings.max_chars_per_item, (
        "the chunker's hard cap is no longer comfortably inside the provider's per-item cap; "
        "the ValueError above has become reachable from the API and needs a typed error"
    )
