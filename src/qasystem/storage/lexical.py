"""FTS5 lexical search (plan.md §6, P5).

Two things make this path safe:

* **Our tokenizer's output is what is indexed** — ZWNJ split components included — so
  SQLite's own tokenizer never has to understand Persian, and a ZWNJ document matches
  a spaced query (D8).
* **Every query joins to ``eligible_chunks``.** Staging, superseded and deleted rows can
  be present in the index and are still unreachable, which is what makes cleanup
  best-effort without becoming a correctness problem (I1, I2).

User text is never passed as FTS5 syntax: ``fts_query_terms`` emits one quoted term per
token, and a token cannot contain a quote.
"""

from __future__ import annotations

from dataclasses import dataclass

from qasystem.storage.sqlite_store import SqliteStore
from qasystem.text.tokenize import fts_query_terms


@dataclass(frozen=True)
class LexicalHit:
    """One lexical match: the FTS row's ``bm25`` plus a rank and a normalized score.

    ``rank`` is 1-based, matching RRF's ``k + rank`` and the dense arm. A 0-based rank is
    one off-by-one away from dividing by ``k`` for the single best hit in the set, which is
    exactly the artefact reciprocal rank fusion exists to remove.
    """

    chunk_id: str
    bm25: float
    lexical_score: float  # 1.0 for the best hit in this result set
    rank: int  # 1-based


class LexicalIndex:
    """``bm25`` ranking over the FTS table, always filtered by eligibility."""

    def __init__(self, store: SqliteStore) -> None:
        self._store = store

    def search(self, query: str, *, limit: int = 30) -> list[LexicalHit]:
        """Best-first matches for ``query``, or ``[]``.

        An empty result covers both "nothing matched" and "nothing searchable": a
        blank question and a stopword-only question both yield no terms, and both must
        return nothing rather than the whole corpus.
        """
        terms = fts_query_terms(query)
        if not terms or limit <= 0:
            return []
        rows = self._store.eligible_chunks_fts(terms, limit)
        # bm25 is negative and lower is better, so the best hit is the minimum. Dividing
        # by it puts the top hit at exactly 1.0 and the rest in (0, 1].
        best = min((row["bm25"] for row in rows), default=0.0)
        return [
            LexicalHit(
                chunk_id=row["chunk_id"],
                bm25=float(row["bm25"]),
                lexical_score=(float(row["bm25"]) / best) if best < 0 else 0.0,
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]
