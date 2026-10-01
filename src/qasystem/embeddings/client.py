"""Async HTTP embedding client (plan.md P4, C3-C6).

The only external service is embedding-only — there is no chat endpoint — so this
module's whole job is: discover the model, probe its dimension, batch inputs, and
turn responses back into vectors **in the caller's order**.

Three §2.1 measurements shape the code, each pinned by a test here:

- **``index`` is not trustworthy.** Gemini returns ``index: [0,0,0,0]``, not a
  permutation, so trusting it would silently mislabel every vector. ``index`` is used
  only when it is a genuine permutation of ``0..n-1``; otherwise response order wins
  and a warning is logged.
- **A single item has its own cap.** ``MAX_CHARS_PER_ITEM`` exists because a 45 000-char
  item fails with ``maximum context length is 8192 token`` while 40 000 succeeds.
- **``Retry-After`` is absent on 429**, so capped exponential backoff with jitter is the
  real path; the header is still honored if a provider ever sends it.

No fallback to another model, ever (C4). A failure is a typed error, not a different
embedding.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import httpx

from qasystem.config import Settings
from qasystem.embeddings.rate_limit import RateLimiter
from qasystem.errors import ConfigError, EmbeddingAuthError, EmbeddingUnavailableError
from qasystem.logging_setup import scrub

logger = logging.getLogger(__name__)

# §2.1: the provider states the allowed models inside the 403 message.
_ALLOWED_RE = re.compile(r"Allowed:\s*(.+)", re.IGNORECASE)
_RETRY_BASE_S = 0.5
_RETRY_CAP_S = 30.0


def plan_batches(
    texts: Sequence[str],
    *,
    max_chars_per_request: int,
    max_items: int,
    max_chars_per_item: int,
) -> list[list[str]]:
    """Split ``texts`` into request-sized groups obeying all three bounds.

    The per-item cap is checked first and raises: a single oversized item cannot be made
    to fit, and truncating it would embed a vector for text nobody cited (I6).
    """
    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in texts:
        if len(text) > max_chars_per_item:
            raise ValueError(
                f"item of {len(text)} chars exceeds MAX_CHARS_PER_ITEM={max_chars_per_item}; "
                "the model context cannot hold it (lower CHUNK_HARD_MAX_TOKENS)"
            )
        fits = len(current) < max_items and (
            not current or chars + len(text) <= max_chars_per_request
        )
        if current and not fits:
            batches.append(current)
            current, chars = [], 0
        current.append(text)
        chars += len(text)
    if current:
        batches.append(current)
    return batches


def _is_permutation(indices: Sequence[Any], expected: int) -> bool:
    return (
        len(indices) == expected
        and all(isinstance(index, int) for index in indices)
        and sorted(indices) == list(range(expected))
    )


class EmbeddingClient:
    """``Embedder`` over ``POST /embeddings``, with startup discovery and bounded retries."""

    def __init__(
        self,
        settings: Settings,
        *,
        limiter: RateLimiter | None = None,
        http: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        # Filled in by start(); empty until then, so nothing can use a wrong dimension.
        self.model_id = ""
        self.dimension = 0
        self.available_models: tuple[str, ...] = ()
        self.limiter = limiter if limiter is not None else RateLimiter(settings.rate_limit_per_min)
        self._sleep = sleep
        self._http = http or httpx.AsyncClient(
            base_url=settings.embedding_base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.api_key()}"},
            timeout=settings.request_timeout_s,
        )
        self._owns_http = http is None

    async def __aenter__(self) -> EmbeddingClient:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def start(self) -> EmbeddingClient:
        """Verify the configured model exists, then probe its dimension with one call."""
        if not self.settings.embedding_model:
            raise ConfigError("EMBEDDING_MODEL is required")
        self.available_models = await self._list_models()
        if self.settings.embedding_model not in self.available_models:
            raise EmbeddingAuthError(
                f"model {self.settings.embedding_model!r} is not available. "
                f"Allowed: {', '.join(self.available_models) or 'none'}",
                details={"available_models": list(self.available_models)},
            )
        self.model_id = self.settings.embedding_model
        self.dimension = len((await self.embed(["dimension probe"]))[0])
        expected = self.settings.embedding_dimension
        if expected is not None and expected != self.dimension:
            raise ConfigError(
                f"EMBEDDING_DIMENSION={expected} but {self.model_id} returned "
                f"{self.dimension}; refusing to mix dimensions (I9)"
            )
        logger.info(
            "embedding model ready: model=%s dimension=%d available=%s",
            self.model_id,
            self.dimension,
            list(self.available_models),
        )
        return self

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per input, in input order. Raises typed errors, never falls back."""
        if not texts:
            return []
        out: list[list[float]] = []
        for batch in plan_batches(
            texts,
            max_chars_per_request=self.settings.max_chars_per_request,
            max_items=self.settings.max_items_per_batch,
            max_chars_per_item=self.settings.max_chars_per_item,
        ):
            out.extend(await self._post_embeddings(batch))
        return out

    async def _list_models(self) -> tuple[str, ...]:
        payload = await self._get_json("models")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise EmbeddingUnavailableError("GET /models returned an unrecognised body")
        return tuple(str(item.get("id", "")) for item in data if isinstance(item, dict))

    async def _get_json(self, path: str) -> Mapping[str, Any]:
        try:
            response = await self._http.get(path)
        except httpx.HTTPError as exc:
            raise EmbeddingUnavailableError(f"GET {path} failed: {type(exc).__name__}") from None
        if response.status_code >= 400:
            raise self._to_error(response)
        try:
            payload = response.json()
        except ValueError:
            raise EmbeddingUnavailableError(f"GET {path} returned a non-JSON body") from None
        if not isinstance(payload, dict):
            raise EmbeddingUnavailableError(f"GET {path} returned an unexpected JSON shape")
        return payload

    async def _post_embeddings(self, batch: Sequence[str]) -> list[list[float]]:
        body = {"model": self.model_id, "input": list(batch)}
        for attempt in range(self.settings.max_retries + 1):
            await self.limiter.acquire()
            try:
                response = await self._http.post("embeddings", json=body)
            except httpx.TimeoutException as exc:
                if attempt < self.settings.max_retries:
                    await self._backoff(attempt, None)
                    continue
                raise EmbeddingUnavailableError(
                    f"POST /embeddings timed out after {attempt + 1} attempts"
                ) from exc
            except httpx.HTTPError as exc:
                if attempt < self.settings.max_retries:
                    await self._backoff(attempt, None)
                    continue
                raise EmbeddingUnavailableError(
                    f"POST /embeddings failed: {type(exc).__name__}"
                ) from None
            if response.status_code < 400:
                return self._reassemble(response, len(batch))
            if not _retryable(response.status_code) or attempt >= self.settings.max_retries:
                raise self._to_error(response)
            await self._backoff(attempt, response.headers.get("Retry-After"))
        raise EmbeddingUnavailableError("POST /embeddings failed")  # pragma: no cover

    def _reassemble(self, response: httpx.Response, expected: int) -> list[list[float]]:
        """Restore input order, trusting ``index`` only when it is a permutation."""
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingUnavailableError(
                f"expected {expected} vectors, got "
                f"{len(data) if isinstance(data, list) else 'no list'}"
            )
        indices = [item.get("index") for item in data]
        if _is_permutation(indices, expected):
            by_index = {item["index"]: item for item in data}
            ordered = [by_index[i] for i in range(expected)]
        else:
            logger.warning(
                "provider returned a non-permutation index field (%s); "
                "using response order instead",
                indices,
            )
            ordered = data
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingUnavailableError("response item has no embedding list")
            vectors.append([float(value) for value in vector])
        # Every vector in a batch shares one dimension: a ragged response would put
        # unusable rows in Chroma. Before the probe, self.dimension is 0 and the
        # batch's own first vector is the reference.
        width = self.dimension or len(vectors[0])
        if any(len(vector) != width for vector in vectors):
            raise EmbeddingUnavailableError(
                f"inconsistent vector dimension: expected {width}, "
                f"got {[len(vector) for vector in vectors]}"
            )
        return vectors

    def _to_error(self, response: httpx.Response) -> Exception:
        """Map a provider response to a typed error. Never carries the token (I8)."""
        detail = self._error_detail(response)
        if response.status_code in (401, 403):
            available = _allowed_models(detail)
            return EmbeddingAuthError(
                f"embedding API rejected the request ({response.status_code}): {detail}",
                details={"available_models": available},
            )
        return EmbeddingUnavailableError(f"embedding API returned {response.status_code}: {detail}")

    async def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(_RETRY_CAP_S, _RETRY_BASE_S * 2**attempt)
        if retry_after:
            # A date-formatted header is not a float; backoff is the honest fallback.
            with contextlib.suppress(ValueError):
                delay = min(_RETRY_CAP_S, max(0.0, float(retry_after)))
        await self._sleep(delay * (0.5 + random.random() / 2))

    def _error_detail(self, response: httpx.Response) -> str:
        """The provider's own message, which is where the model list and caps live (§2.1).

        Scrubbed before it leaves the client: a provider that echoes the Authorization
        header must not turn our error body into an I8 leak.
        """
        try:
            payload = response.json()
        except ValueError:
            return self._scrub(response.text[:200])
        detail = (
            payload["error"][:200]
            if isinstance(payload, dict) and isinstance(payload.get("error"), str)
            else response.text[:200]
        )
        return self._scrub(str(detail))

    def _scrub(self, text: str) -> str:
        key = self.settings.embedding_api_key
        return scrub((key.get_secret_value() if key else "",), text)


def _retryable(status: int) -> bool:
    """429 and 5xx only. §2.1: Retry-After is absent, so backoff must carry the load."""
    return status == 429 or status >= 500


def _allowed_models(detail: str) -> list[str]:
    match = _ALLOWED_RE.search(detail)
    if match is None:
        return []
    return [item.strip() for item in match.group(1).split(",") if item.strip()]
