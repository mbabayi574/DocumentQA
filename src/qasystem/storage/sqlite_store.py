"""SQLite repository (plan.md §6, P5).

One connection, short-lived transactions, one obvious boundary per method. The store
owns every write: a caller cannot reach a cursor and leave a half-applied change behind.

Two rules the code exists to enforce:

* **Publish is one transaction** that flips ``documents.current_version``. There is no
  second "is active" flag, so eligibility is derived, never remembered (I1).
* **A failed stage leaves nothing behind.** ``stage_version`` writes the version row,
  the chunk rows and the FTS rows in one transaction, so I3's "previous version fully
  queryable" needs no compensating writes.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import struct
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from qasystem.domain.models import Chunk
from qasystem.embeddings.caching import input_hash
from qasystem.text.tokenize import tokenize

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def chunk_id_for(doc_id: str, version: int, ordinal: int) -> str:
    """The one place a chunk id is built; it doubles as the Chroma vector id (§4.1)."""
    return f"{doc_id}:v{version}:{ordinal}"


class SqliteStore:
    """Documents, versions, chunks, FTS, embedding cache, ingest log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the connection is built in the lifespan and used
        # from request threads. Safe here because of the two guarantees this project already
        # makes: L2's data-directory lock means one *process* owns the file, and the
        # application write lock means one coroutine at a time mutates it. Without this,
        # building a store anywhere but the request thread raises ProgrammingError (D49).
        # Python's sqlite3 is compiled SERIALIZED, so the C layer still serialises access.
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA busy_timeout = 5000")
        self._migrate()

    def close(self) -> None:
        """Close the connection. Further use raises, which is the point."""
        self._db.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def fts5_available() -> bool:
        """A startup check, not an assumption: FTS5 may be compiled out."""
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            return True
        except sqlite3.OperationalError:
            return False
        finally:
            probe.close()

    # ------------------------------------------------------------- plumbing

    def _migrate(self) -> None:
        self._db.executescript(
            resources.files("qasystem.storage").joinpath("schema.sql").read_text(encoding="utf-8")
        )
        current = self._db.execute("PRAGMA user_version").fetchone()[0]
        if current > SCHEMA_VERSION:
            raise ValueError(
                f"database is at schema v{current}, this build understands v{SCHEMA_VERSION}"
            )
        if current < SCHEMA_VERSION:
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @contextlib.contextmanager
    def _txn(self) -> Iterator[None]:
        """One explicit transaction.

        The connection is in autocommit mode, where ``with self._db:`` is a no-op: it
        would commit every statement separately and a failure halfway through would
        leave exactly the partial rows that I3 forbids. BEGIN/COMMIT is spelled out so
        the boundary is unambiguous.
        """
        self._db.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def _rows(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self._db.execute(sql, params)]

    def _one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        row = self._db.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def execute_script(self, sql: str) -> None:
        """Escape hatch for tests and migrations; not a repository method."""
        self._db.executescript(sql)

    def in_transaction(self) -> bool:
        """True while a ``_txn`` block is open."""
        return bool(self._db.in_transaction)

    def journal_mode(self) -> str:
        """The connection's journal mode; ``wal`` is required, not optional."""
        return str(self._db.execute("PRAGMA journal_mode").fetchone()[0])

    def foreign_keys_enabled(self) -> bool:
        """True when FK enforcement is on, so a bad row cannot be written."""
        return bool(self._db.execute("PRAGMA foreign_keys").fetchone()[0])

    def schema_version(self) -> int:
        """``PRAGMA user_version``, the migration marker."""
        return int(self._db.execute("PRAGMA user_version").fetchone()[0])

    # ------------------------------------------------------------- documents

    def begin_staging(
        self,
        doc_id: str,
        title: str,
        source_name: str,
        doc_format: str,
        content_hash: str,
        parsed_hash: str,
        language_hint: str,
    ) -> int:
        """Create or refresh the document row and allocate the next version number.

        One transaction, so two concurrent ingestions cannot be handed the same
        version (plan.md P6 rule 2). ``last_version`` only ever increases, which is
        what makes a delete-then-re-add keep its history.
        """
        stamp = _now()
        with self._txn():
            self._db.execute(
                """
                INSERT INTO documents (doc_id, title, source_name, format, status,
                                       current_version, last_version, content_hash,
                                       parsed_hash, language_hint, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'active', NULL, 0, ?, ?, ?, ?, ?)
                ON CONFLICT (doc_id) DO UPDATE SET
                    title = excluded.title,
                    source_name = excluded.source_name,
                    format = excluded.format,
                    content_hash = excluded.content_hash,
                    parsed_hash = excluded.parsed_hash,
                    language_hint = excluded.language_hint,
                    updated_at = excluded.updated_at
                """,
                (
                    doc_id,
                    title,
                    source_name,
                    doc_format,
                    content_hash,
                    parsed_hash,
                    language_hint,
                    stamp,
                    stamp,
                ),
            )
            self._db.execute(
                "UPDATE documents SET last_version = last_version + 1, updated_at = ? "
                "WHERE doc_id = ?",
                (stamp, doc_id),
            )
            version = int(
                self._db.execute(
                    "SELECT last_version FROM documents WHERE doc_id = ?", (doc_id,)
                ).fetchone()[0]
            )
        return version

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """The document row, or ``None`` when the id is unknown."""
        return self._one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))

    def documents(self, *, status: str | None = "active") -> list[str]:
        """Document ids in a status; ``deleted`` returns the tombstones."""
        if status is None:
            return [
                row["doc_id"] for row in self._rows("SELECT doc_id FROM documents ORDER BY doc_id")
            ]
        return [
            row["doc_id"]
            for row in self._rows(
                "SELECT doc_id FROM documents WHERE status = ? ORDER BY doc_id", (status,)
            )
        ]

    # ------------------------------------------------------------- versions

    def stage_version(
        self,
        doc_id: str,
        version: int,
        content_hash: str,
        source_text: str,
        chunks: Sequence[Chunk],
        embed_inputs: Sequence[str],
        *,
        fail_after: str | None = None,
    ) -> None:
        """Write the version row, its chunks and its FTS rows in one transaction.

        ``embed_inputs`` is the exact string that will be embedded for each chunk, in
        ordinal order; the store records only its sha256, which is what I5's cache and
        I10's rebuild key on. ``fail_after`` is the failure-injection seam P6 needs.
        """
        if len(chunks) != len(embed_inputs):
            raise ValueError(
                f"{len(chunks)} chunks but {len(embed_inputs)} embed inputs; they must pair up"
            )
        stamp = _now()
        with self._txn():
            self._db.execute(
                "INSERT INTO document_versions (doc_id, version, content_hash, source_text,"
                " state, created_at) VALUES (?, ?, ?, ?, 'staging', ?)",
                (doc_id, version, content_hash, source_text, stamp),
            )
            self._insert_chunks(doc_id, version, chunks, embed_inputs)
            if fail_after == "chunks":
                raise RuntimeError("injected failure after chunk rows")
            self._insert_fts(doc_id, version, chunks)
            if fail_after == "fts":
                raise RuntimeError("injected failure after fts rows")

    def _insert_chunks(
        self,
        doc_id: str,
        version: int,
        chunks: Sequence[Chunk],
        embed_inputs: Sequence[str],
    ) -> None:
        self._db.executemany(
            """INSERT INTO chunks (chunk_id, doc_id, doc_version, ordinal, text,
                   char_start, char_end, chunk_hash, embed_input_hash, section_path_json,
                   page_start, page_end, line_start, line_end, language)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    chunk_id_for(doc_id, version, chunk.ordinal),
                    doc_id,
                    version,
                    chunk.ordinal,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.chunk_hash,
                    input_hash(embed_input),
                    json.dumps(list(chunk.section_path), ensure_ascii=False),
                    chunk.page_start,
                    chunk.page_end,
                    chunk.line_start,
                    chunk.line_end,
                    chunk.language,
                )
                for chunk, embed_input in zip(chunks, embed_inputs, strict=True)
            ],
        )

    def _insert_fts(self, doc_id: str, version: int, chunks: Sequence[Chunk]) -> None:
        self._db.executemany(
            "INSERT INTO chunks_fts (chunk_id, tokens) VALUES (?, ?)",
            [
                (
                    chunk_id_for(doc_id, version, chunk.ordinal),
                    " ".join(tokenize(chunk.text)),
                )
                for chunk in chunks
            ],
        )

    def publish(self, doc_id: str, version: int, content_hash: str, parsed_hash: str) -> None:
        """The one transaction that makes a version visible (I1).

        Everything before this point is invisible to every retrieval path, so a failure
        anywhere earlier leaves the previous version exactly as it was (I3).
        """
        stamp = _now()
        with self._txn():
            if (
                self._one(
                    "SELECT 1 FROM document_versions WHERE doc_id = ? AND version = ?",
                    (doc_id, version),
                )
                is None
            ):
                raise sqlite3.IntegrityError(
                    f"cannot publish {doc_id} v{version}: no such staged version"
                )
            self._db.execute(
                "UPDATE document_versions SET state = 'superseded' "
                "WHERE doc_id = ? AND state = 'published'",
                (doc_id,),
            )
            self._db.execute(
                "UPDATE document_versions SET state = 'published', published_at = ? "
                "WHERE doc_id = ? AND version = ?",
                (stamp, doc_id, version),
            )
            self._db.execute(
                """UPDATE documents SET current_version = ?, status = 'active',
                       content_hash = ?, parsed_hash = ?, published_at = ?,
                       updated_at = ?, deleted_at = NULL
                   WHERE doc_id = ?""",
                (version, content_hash, parsed_hash, stamp, stamp, doc_id),
            )

    def mark_deleted(self, doc_id: str) -> None:
        """I2's whole guarantee: one transaction, then the document is unretrievable.

        The row stays as a tombstone so ``last_version`` keeps increasing and a
        re-add cannot collide with the deleted version's chunk ids.
        """
        stamp = _now()
        with self._txn():
            self._db.execute(
                "UPDATE documents SET status = 'deleted', current_version = NULL, "
                "deleted_at = ?, updated_at = ? WHERE doc_id = ?",
                (stamp, stamp, doc_id),
            )

    def mark_failed(self, doc_id: str, version: int) -> None:
        """Record that a version died before publish (P6 rule 4)."""
        with self._txn():
            self._db.execute(
                "UPDATE document_versions SET state = 'failed' WHERE doc_id = ? AND version = ?",
                (doc_id, version),
            )

    def purge_version(self, doc_id: str, version: int) -> None:
        """Best-effort cleanup of a superseded or failed version's rows."""
        with self._txn():
            self._db.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id = ? AND doc_version = ?)",
                (doc_id, version),
            )
            self._db.execute(
                "DELETE FROM chunks WHERE doc_id = ? AND doc_version = ?", (doc_id, version)
            )
            self._db.execute(
                "DELETE FROM document_versions WHERE doc_id = ? AND version = ?",
                (doc_id, version),
            )

    def purge_version_chunks(self, doc_id: str, version: int) -> None:
        """Drop a version's chunk and FTS rows but keep the version row.

        Used for a version that died before publish: the rows are useless, but the row
        recording that this version number was allocated and failed is the audit trail.
        ``ingest_log`` carries the error code; this carries the identity.
        """
        with self._txn():
            self._db.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM chunks WHERE doc_id = ? AND doc_version = ?)",
                (doc_id, version),
            )
            self._db.execute(
                "DELETE FROM chunks WHERE doc_id = ? AND doc_version = ?", (doc_id, version)
            )

    def version_state(self, doc_id: str, version: int) -> str | None:
        """``staging``/``published``/``superseded``/``failed``, or ``None`` if absent."""
        row = self._one(
            "SELECT state FROM document_versions WHERE doc_id = ? AND version = ?",
            (doc_id, version),
        )
        return str(row["state"]) if row else None

    def versions(self, doc_id: str) -> list[dict[str, Any]]:
        """Every version ever allocated for a document, in order."""
        return self._rows(
            "SELECT doc_id, version, state, created_at, published_at FROM document_versions "
            "WHERE doc_id = ? ORDER BY version",
            (doc_id,),
        )

    def source_text(self, doc_id: str, version: int) -> str | None:
        """One version's parsed text, so I6 is checkable without re-parsing."""
        row = self._one(
            "SELECT source_text FROM document_versions WHERE doc_id = ? AND version = ?",
            (doc_id, version),
        )
        return str(row["source_text"]) if row else None

    def chunk_count(self, doc_id: str, version: int) -> int:
        """Chunk rows a version has; zero after a purge."""
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM chunks WHERE doc_id = ? AND doc_version = ?",
                (doc_id, version),
            ).fetchone()[0]
        )

    def lexical_count(self, chunk_id: str) -> int:
        """FTS rows for a chunk id; proves cleanup reached both tables."""
        return int(
            self._db.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()[0]
        )

    def chunks_for_version(self, doc_id: str, version: int) -> list[dict[str, Any]]:
        """One version's chunks with hashes; the input to P6's diff."""
        return self._rows(
            "SELECT chunk_id, doc_id, doc_version, ordinal, text, char_start, char_end,"
            " chunk_hash, embed_input_hash, section_path_json, page_start, page_end,"
            " line_start, line_end, language FROM chunks "
            "WHERE doc_id = ? AND doc_version = ? ORDER BY ordinal",
            (doc_id, version),
        )

    # ------------------------------------------------------------- eligibility

    def eligible_chunk_ids(self) -> list[str]:
        """Every retrievable chunk id. Tests and /ready use it."""
        return [
            row["chunk_id"]
            for row in self._rows("SELECT chunk_id FROM eligible_chunks ORDER BY chunk_id")
        ]

    def eligible_chunks(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read chunks through ``eligible_chunks`` and nothing else (I1, I2).

        Dense ids and the API's document/language filters all land here, so there is
        exactly one place eligibility is decided and one statement that also returns
        the text a citation will quote.
        """
        sql = "SELECT * FROM eligible_chunks WHERE 1 = 1"
        params: list[Any] = []
        if chunk_ids is not None:
            if not chunk_ids:
                return []
            sql += f" AND chunk_id IN ({_placeholders(len(chunk_ids))})"
            params.extend(chunk_ids)
        if doc_ids is not None:
            if not doc_ids:
                return []
            sql += f" AND doc_id IN ({_placeholders(len(doc_ids))})"
            params.extend(doc_ids)
        if language is not None:
            sql += " AND language = ?"
            params.append(language)
        return self._rows(sql + " ORDER BY doc_id, doc_version, ordinal", params)

    def has_eligible_chunks(self) -> bool:
        """Whether anything at all is searchable.

        Read only on the path that has nothing to answer with, so ``empty_knowledge_base``
        is distinguishable from ``no_relevant_content`` without costing the happy path a
        full id scan.
        """
        return self._one("SELECT 1 AS present FROM eligible_chunks LIMIT 1") is not None

    def expected_vector_ids(self) -> list[str]:
        """Every chunk row, staging included, so reconcile sees an interrupted ingest."""
        return [
            row["chunk_id"] for row in self._rows("SELECT chunk_id FROM chunks ORDER BY chunk_id")
        ]

    def chunks_need_vectors(self) -> list[dict[str, Any]]:
        """The id, location and embed-input hash of every chunk row, staging included.

        ``rebuild`` reads this to restore the dense index from the cache (I10).
        """
        return self._rows(
            "SELECT chunk_id, doc_id, doc_version, ordinal, embed_input_hash FROM chunks "
            "ORDER BY chunk_id"
        )

    def staging_versions(self) -> list[dict[str, Any]]:
        """Versions left in ``staging``: an interrupted ingest that reconcile must fail (P6)."""
        return self._rows(
            "SELECT doc_id, version, created_at FROM document_versions "
            "WHERE state = 'staging' ORDER BY doc_id, version"
        )

    def vector_ids_for(self, doc_id: str, version: int) -> list[str]:
        """The Chroma ids one version owns, so cleanup needs no extra bookkeeping."""
        return [
            row["chunk_id"]
            for row in self._rows(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? AND doc_version = ? ORDER BY ordinal",
                (doc_id, version),
            )
        ]

    def vector_ids_for_document(self, doc_id: str) -> list[str]:
        """Every Chroma id a document owns, across all its versions."""
        return [
            row["chunk_id"]
            for row in self._rows(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY doc_version, ordinal",
                (doc_id,),
            )
        ]

    def eligible_chunks_fts(self, match_expression: str, limit: int) -> list[dict[str, Any]]:
        """Top ``limit`` FTS rows by ``bm25``, joined to ``eligible_chunks``.

        The join is what enforces I1 and I2 on this path. ``match_expression`` must
        already be a quoted, ``OR``-joined expression from ``fts_query_terms`` — this
        method does not escape it, because the only caller builds it safely.
        """
        return self._rows(
            "SELECT f.chunk_id AS chunk_id, bm25(chunks_fts) AS bm25 "
            "FROM chunks_fts f JOIN eligible_chunks e ON e.chunk_id = f.chunk_id "
            "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match_expression, limit),
        )

    # ------------------------------------------------------------- embedding cache

    def get_embeddings(self, model_id: str, input_hashes: Sequence[str]) -> dict[str, list[float]]:
        """Vectors for the hashes that exist. A miss is absent, never a zero vector."""
        if not input_hashes:
            return {}
        sql = (
            f"SELECT input_hash, vector FROM embedding_cache WHERE model_id = ? "
            f"AND input_hash IN ({_placeholders(len(input_hashes))})"
        )
        rows = self._rows(sql, [model_id, *input_hashes])
        return {row["input_hash"]: _unpack(row["vector"]) for row in rows}

    def put_embeddings(self, model_id: str, rows: Iterable[tuple[str, Sequence[float]]]) -> None:
        """Store vectors keyed by ``(input_hash, model_id)``; an upsert."""
        stamp = _now()
        with self._txn():
            self._db.executemany(
                "INSERT INTO embedding_cache (input_hash, model_id, dim, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (input_hash, model_id) DO UPDATE SET "
                "vector = excluded.vector, dim = excluded.dim",
                [(h, model_id, len(v), _pack(v), stamp) for h, v in rows],
            )

    def cached_dimensions(self, model_id: str) -> set[int]:
        """Dimensions this model has cached, so I9 can spot a change."""
        return {
            int(row["dim"])
            for row in self._rows(
                "SELECT DISTINCT dim FROM embedding_cache WHERE model_id = ?", (model_id,)
            )
        }

    # ------------------------------------------------------------- ingest log

    def log_ingest(
        self,
        doc_id: str,
        action: str,
        prev_version: int | None,
        new_version: int | None,
        *,
        chunks_added: int = 0,
        chunks_reused: int = 0,
        chunks_removed: int = 0,
        embed_requests: int = 0,
        duration_ms: int | None = None,
        status: str,
        error_code: str | None = None,
    ) -> None:
        """Append one ingest outcome. Counts and timings only, never text."""
        with self._txn():
            self._db.execute(
                """INSERT INTO ingest_log (doc_id, action, prev_version, new_version,
                       chunks_added, chunks_reused, chunks_removed, embed_requests,
                       duration_ms, status, error_code, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    action,
                    prev_version,
                    new_version,
                    chunks_added,
                    chunks_reused,
                    chunks_removed,
                    embed_requests,
                    duration_ms,
                    status,
                    error_code,
                    _now(),
                ),
            )

    def ingest_log(self, doc_id: str) -> list[dict[str, Any]]:
        """A document's ingest history, oldest first."""
        return self._rows("SELECT * FROM ingest_log WHERE doc_id = ? ORDER BY id", (doc_id,))


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes | memoryview) -> list[float]:
    data = bytes(blob)
    return list(struct.unpack(f"<{len(data) // 4}f", data))
