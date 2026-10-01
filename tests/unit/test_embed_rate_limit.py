"""P4: the rate limiter stays at or under the configured quota under a burst."""

from __future__ import annotations

import pytest

from qasystem.embeddings.rate_limit import RateLimiter


class FakeClock:
    """A clock the test drives by hand; ``sleep`` is the only thing that moves it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.now += seconds


def limiter(per_min: int = 120) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    return RateLimiter(per_min, clock=clock, sleep=clock.sleep), clock


def test_per_min_must_be_positive() -> None:
    with pytest.raises(ValueError, match="per_min"):
        RateLimiter(0, clock=FakeClock())


async def test_a_burst_up_to_the_quota_is_never_delayed() -> None:
    """§2.1 measured 120 rapid calls succeeding, so capacity is a whole minute."""
    gate, clock = limiter(120)
    for _ in range(120):
        await gate.acquire()
    assert clock.slept == 0.0
    assert gate.waits == 0


async def test_sustained_traffic_is_throttled_to_the_configured_rate() -> None:
    """180 calls at 120/min must cost 30s of waiting, not zero."""
    gate, clock = limiter(120)
    for _ in range(180):
        await gate.acquire()
    assert 29.0 <= clock.slept < 31.0  # 60 extra calls at 2 tokens/s
    assert gate.waits > 0


async def test_an_idle_minute_restores_full_capacity() -> None:
    gate, clock = limiter(10)
    for _ in range(10):
        await gate.acquire()
    clock.now += 60.0
    for _ in range(10):
        await gate.acquire()
    assert clock.slept == 0.0


async def test_tokens_are_capped_so_idling_banks_no_extra_credit() -> None:
    """An hour idle must not buy 3600 calls of burst."""
    gate, clock = limiter(10)
    clock.now += 3600.0
    for _ in range(25):
        await gate.acquire()
    assert clock.slept > 0
