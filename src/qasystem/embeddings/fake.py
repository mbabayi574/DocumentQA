"""Offline, deterministic embedder for tests and demo mode.

Vectors are a hashed bag-of-tokens, L2-normalised: identical text always gives the
same vector, and texts sharing tokens end up closer than unrelated ones, so dense
retrieval can be exercised end to end without network access. It makes no attempt
at semantic quality — model choice is measured in P9 against the real service.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

DEFAULT_DIMENSION = 32
DEFAULT_MODEL_ID = "fake-embedder"


class FakeEmbedder:
    """Hash-seeded, order-preserving, L2-normalised embeddings."""

    def __init__(
        self, dimension: int = DEFAULT_DIMENSION, model_id: str = DEFAULT_MODEL_ID
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension
        self.model_id = model_id
        # Network requests issued. The fake has no network, so this equals its call count,
        # which is exactly what the port's `requests` means (see domain/ports.py).
        self.requests = 0
        self.texts_embedded = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.requests += 1
        self.texts_embedded += len(texts)
        return [self._vector(text) for text in texts]

    def reset(self) -> None:
        # Network requests issued. The fake has no network, so this equals its call count,
        # which is exactly what the port's `requests` means (see domain/ports.py).
        self.requests = 0
        self.texts_embedded = 0

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        # Empty text still needs a defined vector, so fall back to hashing "".
        for token in text.lower().split() or [""]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:8], "big") % self.dimension
            vector[slot] += 1.0 if digest[8] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
