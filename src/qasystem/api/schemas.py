"""Request and response models (plan.md P8).

These exist so the OpenAPI schema is the contract rather than a description of one. Every
field the API promises is declared here, including the ones that are honestly ``null`` --
``page`` for a Markdown document, ``lines`` for a PDF, ``debug`` unless it was asked for.
Declaring them optional-but-present is what stops a caller from guessing which is which.

``evidence_score`` is named to avoid ``confidence`` on purpose: it is a deterministic
ordering signal in ``[0, 1]``, not a calibrated probability.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Status = Literal["answered", "insufficient_information"]
Reason = Literal["empty_knowledge_base", "no_relevant_content", "below_threshold"]


class ErrorBody(BaseModel):
    """One error, three fields, for every failure the API can produce."""

    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    """The single error shape. Unknown failures use it too, with ``INTERNAL_ERROR``."""

    error: ErrorBody


class IngestResponse(BaseModel):
    """The result of adding or replacing a document."""

    doc_id: str
    version: int
    status: Literal["created", "updated", "unchanged"]
    chunks_added: int
    chunks_reused: int
    chunks_removed: int
    embed_requests: int
    duration_ms: int


class Section(BaseModel):
    """One entry of a document's outline."""

    path: str
    char_start: int
    char_end: int
    page: int | None = None
    lines: list[int] | None = None


class DocumentSummary(BaseModel):
    """A row of ``GET /documents``."""

    doc_id: str
    title: str
    document: str
    format: str
    current_version: int
    chunk_count: int
    language: str | None = None
    updated_at: str


class DocumentList(BaseModel):
    """A page of documents, with the total so a caller can paginate without guessing."""

    items: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DocumentDetail(BaseModel):
    """One document with its section outline."""

    doc_id: str
    title: str
    document: str
    format: str
    status: str
    current_version: int
    last_version: int
    chunk_count: int
    language: str | None = None
    created_at: str
    updated_at: str
    published_at: str | None = None
    sections: list[Section]


class Segment(BaseModel):
    """A quoted slice. ``text`` is exactly the chunk's ``[start:end]`` (I6)."""

    text: str
    citation_id: int
    chunk_char_start: int
    chunk_char_end: int


class Citation(BaseModel):
    """A traceable reference: document, version, section, location, exact excerpt."""

    id: int
    doc_id: str
    document: str
    doc_version: int
    section: str
    page: int | None = None
    lines: list[int] | None = None
    chunk_id: str
    char_span: list[int]
    excerpt: str
    score: float


class QueryRequest(BaseModel):
    """A question. Every field but the question narrows the search, never widens it."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The question. Whitespace-only questions are rejected.",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    doc_ids: list[str] | None = None
    language: Literal["fa", "en", "mixed"] | None = None
    debug: bool = Field(
        default=False,
        description="Include the gate signals and per-candidate scores. Off by default.",
    )


class QueryResponse(BaseModel):
    """An answer or an honest refusal. ``insufficient_information`` is a 200."""

    model_config = ConfigDict(extra="forbid")

    status: Status
    answer: str
    segments: list[Segment]
    evidence_score: float = Field(
        description="Deterministic evidence signal in [0, 1]. NOT a probability."
    )
    citations: list[Citation]
    reason: Reason | None = None
    debug: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Liveness. Deliberately says nothing about dependencies."""

    status: str


class ReadyResponse(BaseModel):
    """Readiness: every dependency named, so a 503 says *which* one is unhappy."""

    status: str
    sqlite: str
    fts5: bool
    chroma: str
    lock_held: bool
    embedder: str
    model_id: str
    dimension: int
    thresholds_calibrated: bool
