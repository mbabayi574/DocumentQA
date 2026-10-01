"""Hybrid retrieval and the public answer entry point (plan.md §7.1).

```text
question → tokens → dense (embed → Chroma top CANDIDATES_N x OVERFETCH)
                   → lexical (FTS5 bm25 top CANDIDATES_N)
                   → ONE SQLite statement: join the union to eligible_chunks, applying
                     doc_ids/language, returning the text a citation will quote
                   → weighted RRF over the union → token coverage per candidate
                   → gate ──fail──► insufficient_information
                        └─pass──► sentence selection → extractive answer
```

Three properties this ordering buys:

* **One statement decides eligibility.** Chroma and FTS both hand over *ids*; the single
  ``eligible_chunks`` join decides which of them exist and returns the chunk text in that
  same read. There is no window between "is it eligible?" and "here is its text" in which
  a publish could slip in, so I1/I2 hold here for the same reason they hold on ingestion.
* **The gate sees the window; the answer quotes the top k.** A chunk ranked sixth with
  perfect token coverage is the best evidence in the result set, and a gate that only saw
  the top k would refuse a question it could answer. Truncation is an *answer-size*
  decision, made after the gate has judged everything retrieved.
* **Both arms always run.** No branch where a dense miss quietly degrades the system to
  lexical-only, which would present as retrieval quality rather than as a fault.
* **The query is embedded verbatim.** Documents were embedded as
  ``"{title} > {breadcrumb}\\n\\n{raw chunk text}"``, so the question is embedded as the
  raw question -- the same kind of string on both sides. Normalising it would make the two
  inconsistent; that is a P9 experiment, not a free change.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qasystem.answering.extractive import Answer, build_answer, refuse
from qasystem.answering.sentences import query_terms, token_coverage
from qasystem.config import Settings
from qasystem.domain.models import VectorHit
from qasystem.domain.ports import Embedder, VectorStore
from qasystem.errors import InvalidQuestionError
from qasystem.retrieval.fusion import fuse
from qasystem.retrieval.gate import GateSignals, GateVerdict, Thresholds, evaluate
from qasystem.storage.lexical import LexicalHit, LexicalIndex
from qasystem.storage.sqlite_store import SqliteStore
from qasystem.text.language import detect_language

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """An eligible chunk plus every score the gate and the answerer need.

    The raw scores sit next to the fused one on purpose. The fused score is only a
    ranking, and a debugging session that cannot see *why* something ranked first is a
    debugging session that guesses.
    """

    chunk_id: str
    doc_id: str
    doc_version: int
    ordinal: int
    source_name: str
    text: str
    section_path: tuple[str, ...]
    char_start: int
    char_end: int
    page_start: int | None
    page_end: int | None
    line_start: int | None
    line_end: int | None
    language: str
    score: float
    dense_rank: int | None
    lexical_rank: int | None
    similarity: float | None
    bm25: float | None
    lexical_score: float | None
    token_coverage: float


class RetrievalService:
    """Retrieves evidence, and either answers from it or refuses."""

    def __init__(
        self,
        *,
        store: SqliteStore,
        vectors: VectorStore,
        embedder: Embedder,
        thresholds: Thresholds,
        settings: Settings,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._thresholds = thresholds
        self._settings = settings
        self._lexical = LexicalIndex(store)

    # ---------------------------------------------------------------- retrieve

    async def retrieve(
        self,
        question: str,
        *,
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
        limit: int | None = None,
        dense_weight: float | None = None,
        lexical_weight: float | None = None,
    ) -> list[Candidate]:
        """Fused candidates, best first. Every one returned is eligible by construction.

        ``dense_weight``/``lexical_weight`` exist so §9.2 can measure a single-arm
        baseline; the shipped values are the configured ones.
        """
        if not query_terms(question):
            raise InvalidQuestionError("the question contains no searchable terms")
        limit = limit or self._settings.top_k
        dense_weight = self._settings.dense_weight if dense_weight is None else dense_weight
        lexical_weight = self._settings.lexical_weight if lexical_weight is None else lexical_weight

        vector = (await self._embedder.embed([question]))[0]
        lexical_hits = self._lexical.search(question, limit=self._settings.candidates_n)
        dense_hits = self._vectors.query(vector, self._dense_n)

        fused, rows = self._join(
            question, dense_hits, lexical_hits, doc_ids, language, dense_weight, lexical_weight
        )
        # L7: a stale or filtered-out vector must not be able to starve the window. When the
        # vector store had more to give and eligibility dropped a lot of it, ask once more.
        # Bounded, because this is a mitigation, not a loop -- eligibility is decided by the
        # view, and all a bigger n can do is feed that view more ids to accept.
        if len(rows) < self._settings.candidates_n and len(dense_hits) >= self._dense_n:
            wider = self._vectors.query(vector, self._dense_n * self._settings.overfetch)
            if len(wider) > len(dense_hits):
                logger.info(
                    "eligibility dropped the dense window to %d; re-querying with n=%d",
                    len(rows),
                    len(wider),
                )
                dense_hits = wider
                fused, rows = self._join(
                    question, wider, lexical_hits, doc_ids, language, dense_weight, lexical_weight
                )

        return self._candidates(question, fused, rows, dense_hits, lexical_hits, limit)

    def _candidates(
        self,
        question: str,
        fused: Sequence[tuple[str, float]],
        rows: dict[str, dict[str, Any]],
        dense_hits: Sequence[VectorHit],
        lexical_hits: Sequence[LexicalHit],
        limit: int,
    ) -> list[Candidate]:
        """Attach a chunk row and both arms' scores to each fused id."""
        terms = query_terms(question)
        dense_ranks = {hit.id: rank for rank, hit in enumerate(dense_hits, start=1)}
        dense_by_id = {hit.id: hit for hit in dense_hits}
        lexical_by_id = {hit.chunk_id: hit for hit in lexical_hits}
        candidates: list[Candidate] = []
        for chunk_id, score in fused:
            row = rows.get(chunk_id)
            if row is None:
                continue  # ineligible: stale version, deleted document, or filtered out
            dense_hit = dense_by_id.get(chunk_id)
            lexical_hit = lexical_by_id.get(chunk_id)
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    doc_id=row["doc_id"],
                    doc_version=int(row["doc_version"]),
                    ordinal=int(row["ordinal"]),
                    source_name=row["source_name"],
                    text=row["text"],
                    section_path=tuple(json.loads(row["section_path_json"])),
                    char_start=int(row["char_start"]),
                    char_end=int(row["char_end"]),
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    line_start=row["line_start"],
                    line_end=row["line_end"],
                    language=row["language"],
                    score=score,
                    dense_rank=dense_ranks.get(chunk_id),
                    lexical_rank=lexical_hit.rank if lexical_hit else None,
                    similarity=dense_hit.similarity if dense_hit else None,
                    bm25=lexical_hit.bm25 if lexical_hit else None,
                    lexical_score=lexical_hit.lexical_score if lexical_hit else None,
                    token_coverage=token_coverage(question, row["text"], terms),
                )
            )
        return candidates[:limit]

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        language: str | None = None,
        debug: bool = False,
    ) -> Answer:
        """Retrieve, gate, then answer -- or refuse. The gate runs before any text is chosen."""
        question_language = language or _question_language(question)
        # The gate judges the whole retrieved window; only afterwards is it cut to top_k.
        candidates = await self.retrieve(
            question, doc_ids=doc_ids, language=language, limit=self._settings.candidates_n
        )
        if not candidates:
            # Read only when there is nothing to answer with, so the happy path pays nothing.
            reason = (
                "no_relevant_content"
                if self._store.has_eligible_chunks()
                else "empty_knowledge_base"
            )
            return refuse(reason, language=question_language)

        verdict = evaluate(self._signals(candidates), self._thresholds)
        answer = (
            build_answer(
                candidates[: (top_k or self._settings.top_k)],
                question,
                language=question_language,
                max_sentences=self._settings.max_answer_sentences,
                min_sentence_overlap=self._thresholds.min_sentence_overlap,
                evidence_score=_evidence_score(verdict.signals),
            )
            if verdict.passed
            else refuse("below_threshold", language=question_language)
        )
        return _with_debug(answer, verdict, candidates) if debug else answer

    # ---------------------------------------------------------------- internals

    def _join(
        self,
        question: str,
        dense_hits: Sequence[VectorHit],
        lexical_hits: Sequence[LexicalHit],
        doc_ids: Sequence[str] | None,
        language: str | None,
        dense_weight: float,
        lexical_weight: float,
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, Any]]]:
        """Fuse, then resolve eligibility in one statement. Fused is untruncated.

        Truncating before the join is what would let a filtered-out vector cost the query a
        slot, so the fusion is returned whole and the caller slices afterwards.
        """
        dense = {hit.id: (rank, hit.similarity) for rank, hit in enumerate(dense_hits, start=1)}
        lexical = {hit.chunk_id: (hit.rank, hit.lexical_score) for hit in lexical_hits}
        fused = fuse(
            dense,
            lexical,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            k=self._settings.rrf_k,
        )
        rows = {
            row["chunk_id"]: row
            for row in self._store.eligible_chunks(
                chunk_ids=[chunk_id for chunk_id, _ in fused], doc_ids=doc_ids, language=language
            )
        }
        return fused, rows

    def _signals(self, candidates: Sequence[Candidate]) -> GateSignals:
        """The numbers the rule reads, plus the candidate count it needs.

        Coverage is the **maximum** over candidates, not the top-ranked candidate's: the
        question is whether *any* retrieved chunk contains the query's words, and trusting
        the fused rank to have put that chunk first would make the gate depend on the very
        ranking it exists to check.
        """
        top = max(self._settings.top_k, 1)
        dense_top = {
            c.chunk_id for c in candidates if c.dense_rank is not None and c.dense_rank <= top
        }
        lexical_top = {
            c.chunk_id for c in candidates if c.lexical_rank is not None and c.lexical_rank <= top
        }
        union = dense_top | lexical_top
        return GateSignals(
            max_dense=max((c.similarity or 0.0 for c in candidates), default=0.0),
            lexical=max((c.lexical_score or 0.0 for c in candidates), default=0.0),
            token_coverage=max((c.token_coverage for c in candidates), default=0.0),
            overlap=len(dense_top & lexical_top) / len(union) if union else 0.0,
            candidate_count=len(candidates),
        )

    @property
    def _dense_n(self) -> int:
        return self._settings.candidates_n * self._settings.overfetch


def _with_debug(answer: Answer, verdict: GateVerdict, candidates: Sequence[Candidate]) -> Answer:
    """Gate internals are opt-in; a normal response never carries scoring detail."""
    return Answer(
        status=answer.status,
        answer=answer.answer,
        segments=answer.segments,
        citations=answer.citations,
        evidence_score=answer.evidence_score,
        reason=answer.reason,
        language=answer.language,
        debug={
            "gate": verdict.as_dict(),
            "candidates": [
                {
                    "chunk_id": candidate.chunk_id,
                    "score": round(candidate.score, 6),
                    "similarity": candidate.similarity,
                    "bm25": candidate.bm25,
                    "lexical_score": candidate.lexical_score,
                    "token_coverage": round(candidate.token_coverage, 4),
                    "dense_rank": candidate.dense_rank,
                    "lexical_rank": candidate.lexical_rank,
                }
                for candidate in candidates
            ],
        },
    )


def _evidence_score(signals: GateSignals) -> float:
    """A deterministic ordering signal in ``[0, 1]``. Not a probability -- see the module.

    Coverage carries the most weight because §2 measures dense R@1 at 0.67: raw similarity
    is the weakest of the three signals, and a score that mostly tracked similarity would
    rank near-misses above real answers.
    """
    return min(
        1.0,
        max(0.0, 0.4 * signals.token_coverage + 0.3 * signals.max_dense + 0.3 * signals.lexical),
    )


def _question_language(question: str) -> str:
    """The refusal message's language, detected from the question only.

    Detection picks a fixed system string; it is never a routing decision, because one
    collection serves both languages and no translation step exists (§2).
    """
    return "fa" if detect_language(question) == "fa" else "en"
