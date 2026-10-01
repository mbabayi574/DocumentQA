"""Core value objects (plan.md §4).

Pure stdlib: no FastAPI, Chroma, httpx, or sqlite3 imports (enforced by tests).
Every ``char_start``/``char_end`` pair indexes the ``text`` of the object that
carries it, which is what makes I6 mechanically checkable downstream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Language = Literal["fa", "en", "mixed"]
DocumentFormat = Literal["pdf", "txt", "md"]


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
