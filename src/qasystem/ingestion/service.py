"""Ingestion and change management (plan.md P6) -- the highest-risk phase.

The flow is fixed, and every step exists because of a way the naive version fails:

```text
parse → chunk → hash → unchanged? ──yes──► "unchanged", zero embed calls
                             │no
   embed only the inputs the cache does not have
                             ▼
   STAGE   version row + chunk rows + FTS rows, one SQLite transaction
   STAGE   Chroma upsert under "{doc_id}:v{N}:{ordinal}"
   VERIFY  get_existing_ids(expected) == expected
   PUBLISH one SQLite transaction flipping current_version
   CLEANUP superseded rows and vectors, best-effort and separate
```

Three properties do the work:

* **Embedding happens before the lock, and needs no version.** The embedded string is
  ``"{title} > {breadcrumb}\\n\\n{chunk text}"``, so the network-bound part never blocks
  another document's ingestion -- and a document that turns out to be unchanged has cost
  nothing at all (I4).
* **Staging rows are invisible by construction.** ``eligible_chunks`` requires
  ``current_version = doc_version``, so a half-finished ingest is unretrievable without
  anyone remembering to clean up after a crash.
* **Cleanup is best-effort and runs after the publish commit.** It failing may leak
  storage; it can never change what is retrievable (I2), because the view already refuses
  every row of a deleted or superseded document.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

from qasystem.chunking.chunker import Chunker
from qasystem.domain.models import Chunk, VectorItem
from qasystem.domain.ports import DocumentParser, Embedder, VectorStore
from qasystem.errors import DocumentNotFoundError, VectorStoreError
from qasystem.ingestion.diff import DiffPlan, diff_hashes
from qasystem.ingestion.reconcile import ReconcilePlan, plan_reconcile, rebuild
from qasystem.storage.sqlite_store import SqliteStore, chunk_id_for

logger = logging.getLogger(__name__)

Status = Literal["created", "updated", "unchanged"]
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def default_doc_id(filename: str) -> str:
    """Filename stem, lowercased and slugged. Junk names fall back to ``"document"``."""
    stem = filename.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return _SLUG_RE.sub("-", stem.lower()).strip("-") or "document"


def parsed_hash(text: str) -> str:
    """sha256 of the *parsed* text, so a whitespace-only byte change is not a new version."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_input(title: str, section_path: Sequence[str], text: str) -> str:
    """The exact string handed to the embedder (plan.md P3).

    Never stored in the vector store, only hashed into ``embed_input_hash``, so changing
    this template correctly invalidates cached embeddings -- the cache is keyed on this
    string rather than on a template version.
    """
    return f"{title} > {' > '.join(section_path)}\n\n{text}"


@dataclass(frozen=True)
class IngestResult:
    """What one ingestion did, in the terms the API and the ingest log both need."""

    doc_id: str
    version: int
    status: Status
    chunks_added: int
    chunks_reused: int
    chunks_removed: int
    embed_requests: int
    duration_ms: int


class IngestionService:
    """Add, replace, and delete documents with provable version consistency."""

    def __init__(
        self,
        *,
        store: SqliteStore,
        vectors: VectorStore,
        embedder: Embedder,
        parser: DocumentParser,
        chunker: Chunker,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._parser = parser
        self._chunker = chunker
        # ponytail: one global lock, because a single process owns one SQLite file and one
        # Chroma directory (L2/L3). Per-document locks if concurrent ingest throughput matters.
        # ponytail: SQLite and Chroma calls run inline rather than in asyncio.to_thread.
        # They are millisecond-scale and a single process owns both stores (L2/L3), so
        # wrapping 20 call sites buys nothing today. Move them if a single call ever
        # blocks long enough to stall a query, or if a second worker ever appears.
        self._write_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    # -------------------------------------------------------------- ingest

    async def ingest(
        self, data: bytes, filename: str, *, doc_id: str | None = None
    ) -> IngestResult:
        """Add or replace a document. Raises typed errors; never degrades silently."""
        started = time.perf_counter()
        doc_id = doc_id or default_doc_id(filename)
        document = self._parser.parse(data, filename)
        chunks = self._chunker.chunk(document)
        if not chunks:
            raise VectorStoreError(f"{doc_id} produced no chunks; refusing to publish it")
        inputs = [embed_input(document.title, c.section_path, c.text) for c in chunks]
        content = hashlib.sha256(data).hexdigest()
        parsed = parsed_hash(document.text)

        # Outside the lock on purpose: this is the network-bound part and it needs no
        # version number, so it must not serialise against another document's ingestion.
        # `requests` is cumulative on the embedder, so a delta is what this call cost.
        before = _requests(self._embedder)
        try:
            vectors = await self._embedder.embed(inputs)
            requests = _requests(self._embedder) - before
            async with self._write_lock:
                # Re-checked under the lock: a concurrent ingestion may have published
                # this exact text while we were embedding. That wasted work is ours.
                current = self._store.get_document(doc_id)
                if (
                    current is not None
                    and current["status"] == "active"
                    and current["parsed_hash"] == parsed
                ):
                    self._log(
                        doc_id,
                        "unchanged",
                        current["current_version"],
                        current["current_version"],
                        added=0,
                        reused=len(chunks),
                        removed=0,
                        requests=0,
                        elapsed=started,
                        status="ok",
                    )
                    return IngestResult(
                        doc_id=doc_id,
                        version=int(current["current_version"]),
                        status="unchanged",
                        chunks_added=0,
                        chunks_reused=len(chunks),
                        chunks_removed=0,
                        embed_requests=0,
                        duration_ms=_ms(started),
                    )

                plan = self._diff_against_published(doc_id, chunks)
                version = self._store.begin_staging(
                    doc_id,
                    document.title,
                    filename,
                    document.format,
                    content,
                    parsed,
                    _dominant_language(chunks),
                )
                self._stage(doc_id, version, content, document.text, chunks, inputs, vectors)
                self._store.publish(doc_id, version, content, parsed)
                self._purge_superseded(doc_id, version)
        except BaseException as exc:
            self._log(
                doc_id,
                "publish",
                None,
                None,
                added=0,
                reused=0,
                removed=0,
                requests=_requests(self._embedder) - before,
                elapsed=started,
                status="failed",
                error_code=getattr(exc, "code", "INTERNAL_ERROR"),
            )
            raise

        self._log(
            doc_id,
            "publish",
            current["current_version"] if current else None,
            version,
            added=plan.chunks_added,
            reused=plan.chunks_reused,
            removed=plan.chunks_removed,
            requests=requests,
            elapsed=started,
            status="ok",
        )
        return IngestResult(
            doc_id=doc_id,
            version=version,
            status="updated"
            if current is not None and current["status"] == "active"
            else "created",
            chunks_added=plan.chunks_added,
            chunks_reused=plan.chunks_reused,
            chunks_removed=plan.chunks_removed,
            embed_requests=requests,
            duration_ms=_ms(started),
        )

    # -------------------------------------------------------------- delete

    async def delete(self, doc_id: str) -> None:
        """One transaction makes the document unretrievable; the rest is housekeeping.

        Everything after the commit is best-effort by design. The view already refuses
        every row of a deleted document, so a failed purge leaks storage, never evidence
        (I2). The ``documents`` tombstone stays so version numbers keep increasing.
        """
        async with self._write_lock:
            document = self._store.get_document(doc_id)
            if document is None or document["status"] == "deleted":
                raise DocumentNotFoundError(f"no active document with id {doc_id!r}")
            vector_ids = self._store.vector_ids_for_document(doc_id)
            self._store.mark_deleted(doc_id)
            for row in self._store.versions(doc_id):
                self._best_effort(partial(self._store.purge_version, doc_id, row["version"]))
            self._best_effort(partial(self._vectors.delete_ids, vector_ids))
            self._log(
                doc_id,
                "delete",
                document["current_version"],
                None,
                added=0,
                reused=0,
                removed=len(vector_ids),
                requests=0,
                elapsed=time.perf_counter(),
                status="ok",
            )

    # -------------------------------------------------------------- reconcile

    async def reconcile(self) -> ReconcilePlan:
        """Bring the derived index back in line with SQLite (plan.md P6 rule 8).

        SQLite is read to decide and never written *from* the vector store's contents.
        Stale staging versions are failed and purged, unexpected vector ids are removed,
        and a short collection is rebuilt from the embedding cache rather than re-embedded.

        Returns the plan **as found**, which is what a caller needs in order to report
        what drifted -- not one recomputed after the repairs, which would always look
        clean and hide the very thing being reported.
        """
        async with self._write_lock:
            for row in self._store.staging_versions():
                doc_id, version = str(row["doc_id"]), int(row["version"])
                self._store.mark_failed(doc_id, version)
                self._best_effort(
                    partial(self._store.purge_version_chunks, doc_id, version),
                    partial(
                        self._vectors.delete_ids,
                        self._store.vector_ids_for(doc_id, version),
                    ),
                )
                logger.warning("failed stale staging version %s v%d", doc_id, version)

            plan = plan_reconcile(self._store, self._vectors)
            if plan.orphaned:
                self._best_effort(partial(self._vectors.delete_ids, list(plan.orphaned)))
            if plan.missing:
                logger.info("rebuilding %d vectors from the embedding cache", len(plan.missing))
                rebuild(self._store, self._vectors, self.model_id)
            return plan

    # -------------------------------------------------------------- internals

    def _diff_against_published(self, doc_id: str, chunks: Sequence[Chunk]) -> DiffPlan:
        document = self._store.get_document(doc_id)
        version = document["current_version"] if document else None
        prior = (
            [row["chunk_hash"] for row in self._store.chunks_for_version(doc_id, int(version))]
            if version is not None
            else []
        )
        return diff_hashes(prior, [chunk.chunk_hash for chunk in chunks])

    def _stage(
        self,
        doc_id: str,
        version: int,
        content: str,
        source_text: str,
        chunks: Sequence[Chunk],
        inputs: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """Write rows, then vectors, then verify. Any gap aborts before publish (rules 1, 7).

        The verify step is what catches a vector store that reports success while writing
        less than it was given -- publishing then would leave a version whose evidence
        silently fails to retrieve.
        """
        try:
            self._store.stage_version(doc_id, version, content, source_text, chunks, inputs)
            items = [
                VectorItem(
                    id=chunk_id_for(doc_id, version, chunk.ordinal),
                    vector=vector,
                    metadata={
                        "doc_id": doc_id,
                        "doc_version": version,
                        "ordinal": chunk.ordinal,
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            self._vectors.upsert(items)
            expected = {item.id for item in items}
            found = self._vectors.get_existing_ids(sorted(expected))
            if found != expected:
                raise VectorStoreError(
                    f"vector store holds {len(found)} of {len(expected)} expected vectors; "
                    "refusing to publish a partial version"
                )
        except BaseException:
            # The version row stays, marked failed: the number is spent and the audit
            # trail matters. Its chunk rows cannot be, because there may be no evidence
            # of which ones were written.
            self._store.mark_failed(doc_id, version)
            self._best_effort(
                partial(self._store.purge_version_chunks, doc_id, version),
                partial(self._vectors.delete_ids, self._store.vector_ids_for(doc_id, version)),
            )
            raise

    def _purge_superseded(self, doc_id: str, version: int) -> None:
        """Drop every earlier version's rows and vectors. Best-effort, and never fatal."""
        for row in self._store.versions(doc_id):
            if int(row["version"]) < version:
                self._best_effort(
                    partial(self._store.purge_version, doc_id, int(row["version"])),
                    partial(
                        self._vectors.delete_ids,
                        self._store.vector_ids_for(doc_id, int(row["version"])),
                    ),
                )

    def _best_effort(self, *actions: Callable[..., object]) -> None:
        """Run cleanup steps, logging and swallowing failures.

        Cleanup runs after the publish commit, so a failure here may leak rows or vectors.
        That is acceptable: the view decides what is retrievable, so a leak is storage,
        not evidence.
        """
        for action in actions:
            try:
                action()
            except Exception as exc:  # cleanup must not change what is retrievable
                logger.warning("cleanup step failed (%s); continuing", type(exc).__name__)

    def _log(
        self,
        doc_id: str,
        action: str,
        prev_version: int | None,
        new_version: int | None,
        *,
        added: int,
        reused: int,
        removed: int,
        requests: int,
        elapsed: float,
        status: str,
        error_code: str | None = None,
    ) -> None:
        self._store.log_ingest(
            doc_id=doc_id,
            action=action,
            prev_version=int(prev_version) if prev_version is not None else None,
            new_version=int(new_version) if new_version is not None else None,
            chunks_added=added,
            chunks_reused=reused,
            chunks_removed=removed,
            embed_requests=requests,
            duration_ms=_ms(elapsed),
            status=status,
            error_code=error_code,
        )


def _int_or_none(row: Mapping[str, Any] | None, key: str) -> int | None:
    """SQLite columns are ``Any``; this keeps the conversion in one place."""
    if row is None or row[key] is None:
        return None
    return int(row[key])


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _requests(embedder: Embedder) -> int:
    return int(getattr(embedder, "requests", 0))


def _dominant_language(chunks: Sequence[Chunk]) -> str:
    """The document's language label: the most common, ties to English, then alphabetical."""
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.language] = counts.get(chunk.language, 0) + 1
    return max(counts, key=lambda language: (counts[language], language == "en"))
