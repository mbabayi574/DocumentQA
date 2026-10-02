"""Core value objects (plan.md §4).

Pure stdlib: no FastAPI, Chroma, httpx, or sqlite3 imports (enforced by tests).
Every ``char_start``/``char_end`` pair indexes the ``text`` of the object that
carries it, which is what makes I6 mechanically checkable downstream.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Language = Literal["fa", "en", "mixed"]
DocumentFormat = Literal["pdf", "txt", "md"]


def input_hash(text: str) -> str:
    """sha256 of the exact embedded string; the cache key, with no normalization.

    It lives here rather than in `embeddings/` because two layers need it and neither may
    import the other: `embeddings/caching.py` uses it as the cache key, and
    `storage/sqlite_store.py` writes the same value into `chunks.embed_input_hash` so a
    rebuilt index can find the vector again. Two copies of this expression would break I5 and
    I10 silently the day one of them changed (D76).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Section:
    """A logical section of a parsed document."""

    char_start: int
    char_end: int
    section_path: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """Parser output; ``text`` is the "source text" all offsets refer to."""

    title: str
    text: str
    sections: tuple[Section, ...]
    format: DocumentFormat


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of ``ParsedDocument.text`` produced by the chunker."""

    ordinal: int
    char_start: int
    char_end: int
    text: str
    section_path: tuple[str, ...]
    chunk_hash: str  # sha256(canonical(section_path) + normalized text)
    language: Language
    # Optional location fields follow the required ones, so a Chunk can never be
    # built with an empty hash or a silently-defaulted language.
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass(frozen=True)
class VectorItem:
    """A vector plus the diagnostic metadata stored next to it in the vector store."""

    id: str
    vector: Sequence[float]
    metadata: Mapping[str, str | int]


@dataclass(frozen=True)
class VectorHit:
    """A vector-store neighbour; ``similarity`` is ``1 - cosine_distance``."""

    id: str
    similarity: float


@dataclass(frozen=True)
class Candidate:
    """An eligible chunk plus every score the gate and the answerer need.

    It lives here, not in `retrieval/service.py`, because it is the answerer's *input*: the
    retrieval service produces it and the answerer consumes it, so `answering` importing it
    from `retrieval` pointed the dependency the wrong way and closed an import cycle (D76).

    The raw scores sit next to the fused one on purpose. The fused score is only a ranking,
    and a debugging session that cannot see *why* something ranked first is a debugging
    session that guesses.
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
