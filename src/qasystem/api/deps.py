"""The service graph, built once per process (plan.md P8, §4.1 L2/L9).

Everything the HTTP layer needs is constructed here and nowhere else, so a missing
collaborator is a startup failure rather than a surprise on some later request. Two things
this module owns and nothing else is allowed to do:

* **Taking the data-directory lock** (L2). One process owns ``data/`` and the Chroma
  directory, so a second instance must fail fast rather than interleave writes. The lock is
  released on ``close``, which is what lets tests build and tear down repeatedly.
* **Probing the embedding model once** (L9). ``/ready`` reports the model and its dimension
  and must not spend a request per readiness check, so the probe happens here at startup and
  the answer is cached on the graph.

Bringing the system up is ``async`` because it requires a network call to discover the model's
identity -- there is no honest way to open a real provider without one. No Protocol and no
factory: one implementation of each collaborator, and one place that knows how they fit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from qasystem.chunking.chunker import Chunker
from qasystem.config import Settings
from qasystem.domain.ports import Embedder, VectorStore
from qasystem.embeddings.caching import CachingEmbedder, EmbeddingCache
from qasystem.embeddings.client import EmbeddingClient
from qasystem.embeddings.fake import FakeEmbedder
from qasystem.ingestion.service import IngestionService
from qasystem.parsing.registry import ParserRegistry
from qasystem.retrieval.gate import Thresholds, load_thresholds
from qasystem.retrieval.service import RetrievalService
from qasystem.storage.chroma_store import ChromaStore
from qasystem.storage.lock import DataLock
from qasystem.storage.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """The wired graph. Routes read attributes off this and never construct anything."""

    settings: Settings
    store: SqliteStore
    vectors: VectorStore
    embedder: Embedder
    thresholds: Thresholds
    registry: ParserRegistry
    chunker: Chunker
    ingestion: IngestionService
    retrieval: RetrievalService
    lock: DataLock
    model_id: str
    dimension: int
    _client: EmbeddingClient | None = field(default=None, repr=False)

    async def aclose(self) -> None:
        """Release everything, in reverse order of acquisition. Idempotent.

        Async because the provider's HTTP client cannot be closed synchronously, and
        pretending otherwise would leak the connection pool on every restart.
        """
        self.lock.release()
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()
        close = getattr(self.vectors, "close", None)
        if close is not None:
            close()
        self.store.close()


async def build_services(settings: Settings, *, data_dir: str | Path | None = None) -> Services:
    """Construct and validate the whole graph, taking the lock before anything is opened.

    The lock comes first so two processes racing for the same ``data/`` cannot both get as far
    as opening SQLite.

    ``data_dir`` is a test seam that redirects *both* stores at a tmpdir. Without it the
    configured ``SQLITE_PATH`` and ``CHROMA_PATH`` are authoritative -- they are real settings,
    and an earlier version derived both paths from ``data_dir`` instead, which silently ignored
    them (D52).
    """
    resolved = Path(data_dir) if data_dir is not None else settings.data_dir
    sqlite_path = Path(data_dir) / "qasystem.db" if data_dir is not None else settings.sqlite_path
    chroma_path = Path(data_dir) / "chroma" if data_dir is not None else settings.chroma_path
    lock = DataLock(resolved)
    lock.acquire()

    try:
        store = SqliteStore(sqlite_path)
        registry = ParserRegistry(max_upload_mb=settings.max_upload_mb)
        chunker = Chunker(settings)

        # The store *is* the cache: it satisfies `EmbeddingCache` structurally, so no adapter
        # object stands between the two (D75).
        embedder, client, model_id, dimension = await _build_embedder(settings, store)
        # The collection name carries model and dimension (L4) and the adapter asserts them
        # against stored metadata on open (I9), so a model change cannot silently reuse
        # another model's vectors.
        vectors: VectorStore = ChromaStore(chroma_path, model_id=model_id, dimension=dimension)
        vectors.ensure_collection()

        thresholds = load_thresholds(settings.thresholds_path, model_id=model_id)
        logger.info(
            "services ready: model=%s dimension=%d calibrated=%s",
            model_id,
            dimension,
            thresholds.calibrated,
        )

        return Services(
            settings=settings,
            store=store,
            vectors=vectors,
            embedder=embedder,
            thresholds=thresholds,
            registry=registry,
            chunker=chunker,
            ingestion=IngestionService(
                store=store,
                vectors=vectors,
                embedder=embedder,
                parser=registry,
                chunker=chunker,
            ),
            retrieval=RetrievalService(
                store=store,
                vectors=vectors,
                embedder=embedder,
                thresholds=thresholds,
                settings=settings,
            ),
            lock=lock,
            model_id=model_id,
            dimension=dimension,
            _client=client,
        )
    except BaseException:
        lock.release()
        raise


async def _build_embedder(
    settings: Settings, cache: EmbeddingCache
) -> tuple[Embedder, EmbeddingClient | None, str, int]:
    """The embedder plus its identity, probed once.

    The model id and dimension come from the client that did the probe, never from
    configuration, so they are what the provider actually serves. ``CachingEmbedder`` wraps
    the client over the *real* SQLite cache, so I5 and I10 hold through the production wiring
    and not only in a test double.
    """
    if settings.is_fake_provider:
        fake = FakeEmbedder()
        return fake, None, fake.model_id, fake.dimension

    client = EmbeddingClient(settings)
    await client.__aenter__()
    embedder: Embedder = CachingEmbedder(client, cache)
    return embedder, client, client.model_id, client.dimension
