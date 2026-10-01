"""Token-bucket rate limiter for the embedding API (plan.md P4, C5).

The provider allows 120 requests/minute (§2.1), so the bucket's capacity *is* the
per-minute quota: a burst up to the quota costs nothing, and sustained traffic is
smoothed to the configured rate. Measured provider behaviour — 130 rapid calls gave
exactly 120 successes — is why the bucket is not smaller than the quota.

``clock`` and ``sleep`` are injected so the tests can prove the rate with a fake
clock instead of waiting on wall time (P4 gate).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class RateLimiter:
    """One token per request, refilled continuously at ``per_min / 60`` per second."""

    def __init__(
        self,
        per_min: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if per_min <= 0:
            raise ValueError(f"per_min must be positive, got {per_min}")
        self.per_min = per_min
        self.waits = 0
        self._capacity = float(per_min)
        self._tokens = float(per_min)
        self._refill_per_second = per_min / 60.0
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    async def acquire(self) -> None:
        """Block until one token is available, then spend it."""
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            self.waits += 1
            await self._sleep((1.0 - self._tokens) / self._refill_per_second)

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(
            self._capacity,
            self._tokens + max(0.0, now - self._last) * self._refill_per_second,
        )
        self._last = now
