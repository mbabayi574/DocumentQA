"""The extractive answer (plan.md §7.3, §7.4).

Everything here is **selection**, never generation. A segment's text is
``chunk.text[start:end]`` by construction, which is what makes I6 and I7 mechanically
checkable rather than a claim: strip the ``[n]`` markers from the rendered answer and the
remainder is the source slices, concatenated, in source order.

```text
candidates → per chunk: split into sentences, rank by overlap, drop the weak
           → fill the sentence budget in fused-candidate order
           → sort by source position, merge adjacent picks, number the citations
           → render "slice [n]" per segment
```

`evidence_score` is a deterministic ordering signal in ``[0, 1]``, **not** a probability.
The name is deliberate: a field called ``confidence`` invites callers to read it as a
calibrated likelihood, and nothing here is calibrated (``config/thresholds.json`` says
``calibrated: false`` until P9).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from qasystem.answering.sentences import chunk_idf, query_terms, select_sentences
from qasystem.chunking.chunker import Sentence, split_sentences
from qasystem.domain.models import Candidate

AnswerStatus = Literal["answered", "insufficient_information"]

# Fixed system messages. Not evidence, never cited, and never derived from the corpus.
REFUSAL_MESSAGES = {
    "en": "Not enough information in the provided documents.",
    "fa": "اطلاعات کافی در اسناد ارائه‌شده وجود ندارد.",
}


@dataclass(frozen=True)
class Segment:
    """One quoted slice. ``text`` is exactly ``chunk.text[char_start:char_end]`` (I6)."""

    text: str
    citation_id: int
    chunk_id: str
    chunk_char_start: int
    chunk_char_end: int


@dataclass(frozen=True)
class Citation:
    """A traceable reference: document, version, section, location, exact excerpt (§7.4)."""

    id: int
    doc_id: str
    document: str
    doc_version: int
    section: str
    page: int | None
    lines: tuple[int, int] | None
    chunk_id: str
    char_span: tuple[int, int]
    excerpt: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "document": self.document,
            "doc_version": self.doc_version,
            "section": self.section,
            "page": self.page,
            "lines": list(self.lines) if self.lines else None,
            "chunk_id": self.chunk_id,
            "char_span": list(self.char_span),
            "excerpt": self.excerpt,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class Answer:
    """What a query returns. A refusal is a normal ``200``, not an error."""

    status: AnswerStatus
    answer: str
    segments: tuple[Segment, ...] = ()
    citations: tuple[Citation, ...] = ()
    evidence_score: float = 0.0
    reason: str | None = None
    language: str = "en"
    debug: dict[str, Any] | None = None

    def as_dict(self, *, debug: bool = False) -> dict[str, Any]:
        """The API shape (plan.md §8). Gate internals appear only when asked for."""
        return {
            "status": self.status,
            "answer": self.answer,
            "segments": [
                {
                    "text": segment.text,
                    "citation_id": segment.citation_id,
                    "chunk_char_start": segment.chunk_char_start,
                    "chunk_char_end": segment.chunk_char_end,
                }
                for segment in self.segments
            ],
            "evidence_score": round(self.evidence_score, 4),
            "citations": [citation.as_dict() for citation in self.citations],
            "reason": self.reason,
            "debug": self.debug if debug else None,
        }


def refuse(reason: str, *, language: str = "en") -> Answer:
    """The insufficient-evidence response: no segments, no citations, score zero.

    The message follows the question's language and is never cited, because it is system
    text. Presenting the "closest" chunks as citations here is the one thing this function
    must never do -- a citation asserts a document supports the answer.
    """
    return Answer(
        status="insufficient_information",
        answer=REFUSAL_MESSAGES["fa" if language == "fa" else "en"],
        reason=reason,
        language=language,
    )


def build_answer(
    candidates: Sequence[Candidate],
    question: str,
    *,
    language: str,
    max_sentences: int,
    min_sentence_overlap: float,
    evidence_score: float,
) -> Answer:
    """Select sentences across candidates and assemble the answer.

    Candidates arrive fused, best first, and the sentence budget is spent in that order.
    That ordering is a contract, not a detail: the cross-lingual fallback below quotes
    ``candidates[0]`` because it is the chunk the fuser most believes answers the question.
    What stops one long chunk from consuming the whole answer is ``min_sentence_overlap``:
    a chunk's sentences that do not speak to the question are never eligible, so a chunk
    only gets sentences it actually earned.
    """
    terms = query_terms(question)
    if not candidates or not terms or max_sentences <= 0:
        return refuse("below_threshold", language=language)

    idf = chunk_idf([candidate.text for candidate in candidates], terms)
    picks: list[tuple[int, Sentence]] = []
    for rank, candidate in enumerate(candidates):
        remaining = max_sentences - len(picks)
        if remaining <= 0:
            break
        picks.extend(
            (rank, sentence)
            for sentence in select_sentences(
                split_sentences(candidate.text),
                terms,
                idf=idf,
                chunk_score=candidate.score,
                min_overlap=min_sentence_overlap,
                limit=remaining,
            )
        )
    if not picks:
        # Every sentence scored zero, which means the question and the retrieved text share
        # no token at all -- the cross-lingual case. `min_sentence_overlap` is then measuring
        # nothing, and refusing here discards evidence the gate already accepted and that a
        # live run showed is retrievable (D48). Quote the best chunk instead; the padding
        # suppression still applies everywhere else, because this branch only runs when
        # there is nothing to pad with.
        return _quote_best_chunk(candidates, max_sentences, evidence_score, language=language)

    segments, citations = _assemble(picks, candidates)
    if not segments:
        return refuse("below_threshold", language=language)
    return Answer(
        status="answered",
        answer=" ".join(f"{segment.text} [{segment.citation_id}]" for segment in segments),
        segments=tuple(segments),
        citations=tuple(citations),
        evidence_score=evidence_score,
        language=language,
    )


def _quote_best_chunk(
    candidates: Sequence[Candidate],
    max_sentences: int,
    evidence_score: float,
    *,
    language: str,
) -> Answer:
    """The cross-lingual fallback: quote the best chunk, cited precisely.

    ponytail: the sentences are chosen by position, not by relevance, because with no shared
    token there is nothing to rank them on. Bounded at MAX_ANSWER_SENTENCES, and the citation
    still names the document, version, section and page, so a reader who asked in one language
    and reads the other can find the passage immediately. Sentence-level cross-lingual ranking
    needs an embedding per sentence -- P9's `SENTENCE_RERANK`, measured before it lands.
    """
    best = candidates[0]
    sentences = split_sentences(best.text)
    if not sentences:
        return refuse("below_threshold", language=language)
    # A prefix of the chunk's sentences, so one contiguous slice from the first to the last.
    # No per-sentence loop is needed: `split_sentences` leaves only whitespace between
    # consecutive sentences, so a prefix can never contain a gap to bridge.
    chosen = sentences[:max_sentences]
    start, end = chosen[0].start, chosen[-1].end
    segment = Segment(
        text=best.text[start:end],
        citation_id=1,
        chunk_id=best.chunk_id,
        chunk_char_start=start,
        chunk_char_end=end,
    )
    return Answer(
        status="answered",
        answer=f"{segment.text} [1]",
        segments=(segment,),
        citations=(_citation(1, best),),
        evidence_score=evidence_score,
        language=language,
    )


def _assemble(
    picks: Sequence[tuple[int, Sentence]], candidates: Sequence[Candidate]
) -> tuple[list[Segment], list[Citation]]:
    """Order by source position, merge adjacent picks, and number citations in that order."""
    by_chunk: dict[str, list[tuple[int, Sentence]]] = {}
    for rank, sentence in picks:
        by_chunk.setdefault(candidates[rank].chunk_id, []).append((rank, sentence))

    # Source order is (document, version, ordinal, position): the only ordering that means
    # anything when the answer draws on more than one chunk, and it keeps a document's own
    # sentences contiguous so the answer reads as extracted rather than assembled (I7).
    def position(chunk_id: str) -> tuple[str, int, int]:
        rank = by_chunk[chunk_id][0][0]
        candidate = candidates[rank]
        return (candidate.doc_id, candidate.doc_version, candidate.ordinal)

    segments: list[Segment] = []
    citations: list[Citation] = []
    for citation_id, chunk_id in enumerate(sorted(by_chunk, key=position), start=1):
        candidate = candidates[by_chunk[chunk_id][0][0]]
        for start, end in _merged_spans(by_chunk[chunk_id], candidate.text):
            segments.append(
                Segment(
                    # I6 by construction: the slice is taken from the chunk, never retyped.
                    text=candidate.text[start:end],
                    citation_id=citation_id,
                    chunk_id=chunk_id,
                    chunk_char_start=start,
                    chunk_char_end=end,
                )
            )
        citations.append(_citation(citation_id, candidate))
    return segments, citations


def _merged_spans(picks: Sequence[tuple[int, Sentence]], chunk_text: str) -> list[tuple[int, int]]:
    """Collapse sentences that are adjacent in the chunk into one contiguous slice.

    Only whitespace between two picks allows a merge. Merging across real text would quote
    something the selection deliberately left out, which is the opposite of extraction.
    """
    spans: list[tuple[int, int]] = []
    for _, sentence in sorted(picks, key=lambda item: item[1].start):
        if spans and not chunk_text[spans[-1][1] : sentence.start].strip():
            spans[-1] = (spans[-1][0], sentence.end)
        else:
            spans.append((sentence.start, sentence.end))
    return spans


def _citation(citation_id: int, candidate: Candidate) -> Citation:
    """A citation whose excerpt and offsets both point at the same source bytes."""
    return Citation(
        id=citation_id,
        doc_id=candidate.doc_id,
        document=candidate.source_name,
        doc_version=candidate.doc_version,
        section=" > ".join(candidate.section_path),
        page=candidate.page_start,
        lines=(
            (candidate.line_start, candidate.line_end)
            if candidate.line_start is not None and candidate.line_end is not None
            else None
        ),
        chunk_id=candidate.chunk_id,
        char_span=(candidate.char_start, candidate.char_start + len(candidate.text)),
        excerpt=candidate.text,
        score=candidate.score,
    )
