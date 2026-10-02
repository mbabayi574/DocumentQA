"""The HTTP endpoints (plan.md P8).

Routes stay thin on purpose: they translate HTTP into a service call and back, and nothing
else. Every eligibility, version and retrieval decision belongs to a service, so the same
rules hold whether the caller is HTTP, the CLI or a test.

The status codes carry meaning rather than decoration:

* ``POST /documents`` returns 201 created, 200 unchanged, and **409** when the bytes differ
  from an existing active document. A POST that would silently replace content is ambiguous,
  so it refuses and names the verb that is unambiguous.
* ``DELETE`` is 204 with an empty body, and 404 on a repeat -- a tombstone is not a document.
* ``insufficient_information`` is a **200**. It is a normal answer to a question the corpus
  cannot support, not a failure, and the refusal carries no citations.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, File, Form, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response

from qasystem.answering.extractive import Answer
from qasystem.api.deps import Services
from qasystem.api.schemas import (
    DocumentDetail,
    DocumentList,
    ErrorEnvelope,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    Section,
)
from qasystem.errors import DocumentExistsError, DocumentNotFoundError
from qasystem.ingestion.service import IngestResult

logger = logging.getLogger(__name__)

router = APIRouter()


def services_of(request: Request) -> Services:
    """The graph the lifespan built. Present because the app has a lifespan."""
    return request.app.state.services  # type: ignore[no-any-return]


ServicesDep = Annotated[Services, Depends(services_of)]


def error(status_code: int, code: str, message: str, request: Request) -> JSONResponse:
    """The one error shape, built in one place so no route can invent another."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request.state.request_id,
            }
        },
    )


#: Every route that takes a path or query parameter can fail validation, and the app converts
#: that into `VALIDATION_ERROR` inside the one error envelope (`_install_validation_error`).
#: Declaring it once here is what stops FastAPI's default `HTTPValidationError` -- a body the
#: API never returns -- from creeping back into the schema on the routes that forget to.
#: Found by reading what `/docs` advertised; `test_the_schema_never_advertises_an_error_shape_
#: the_api_does_not_return` is what keeps it out.
INVALID_QUERY: dict[int | str, dict[str, Any]] = {422: {"model": ErrorEnvelope}}

#: Swagger UI's "Try it out" prefills from these, and they live on the *body parameter* rather
#: than on the model because OpenAPI places examples beside the schema, not inside it -- on the
#: model they land in `components.schemas` and Swagger never reads them. Without an example the
#: textarea opens as `{"question": ""}`, which the API rejects, so the first click a new user makes
#: is a 422. The three cover the three things a caller actually wants to try: a plain question, a
#: question with gate diagnostics, and a narrowed Persian question.
QUERY_EXAMPLES = [
    {
        "summary": "A question the corpus can answer",
        "description": "Narrows nothing, so retrieval and the gate decide.",
        "value": {"question": "What port does the gateway listen on by default?"},
    },
    {
        "summary": "Gate diagnostics",
        "description": (
            "Returns the gate's own signals with the answer, which is how you see why something "
            "was refused instead of guessing at a threshold."
        ),
        "value": {"question": "What does error code AUR-2291 report?", "debug": True},
    },
    {
        "summary": "Narrowed to one document and one language",
        "description": "Filters only narrow the search; they never widen it.",
        "value": {
            "question": "کد خطای AUR-6107 چه مشکلی را گزارش می‌کند؟",
            "doc_ids": ["incident-runbook"],
            "language": "fa",
        },
    },
]


# ---------------------------------------------------------------- ops


@router.get("/health", response_model=HealthResponse, tags=["ops"], summary="Liveness")
async def health() -> dict[str, str]:
    """Alive. Says nothing about dependencies, because a liveness probe that fails on a
    database outage turns a recoverable problem into a restart loop."""
    return {"status": "ok"}


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={503: {"model": ReadyResponse}},
    tags=["ops"],
    summary="Readiness: every dependency named",
)
async def ready(services: ServicesDep) -> JSONResponse:
    """Report each dependency separately, so a 503 says which one is unhappy.

    The model and its dimension come from the startup probe and are read from cache: L9
    forbids an embedding call per readiness check, or a load balancer polling every two
    seconds would exhaust the provider's rate limit by itself.
    """
    report: dict[str, Any] = {
        "sqlite": "ok",
        "fts5": True,
        "chroma": "ok",
        "lock_held": services.lock.is_held,
        "embedder": "fake" if services.settings.is_fake_provider else "remote",
        "model_id": services.model_id,
        "dimension": services.dimension,
        "thresholds_calibrated": services.thresholds.calibrated,
    }
    try:
        services.store.schema_version()
    except Exception as exc:
        report["sqlite"] = f"error: {type(exc).__name__}"
    if not report["fts5"]:
        report["fts5"] = services.store.fts5_available()
    try:
        services.vectors.ping()
        services.vectors.count()
    except Exception as exc:
        report["chroma"] = f"error: {type(exc).__name__}"

    healthy = (
        report["sqlite"] == "ok"
        and report["fts5"] is True
        and report["chroma"] == "ok"
        and report["lock_held"] is True
    )
    report["status"] = "ready" if healthy else "not_ready"
    return JSONResponse(status_code=200 if healthy else 503, content=report)


# ---------------------------------------------------------------- documents


@router.post(
    "/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"model": IngestResponse, "description": "Identical content: unchanged"},
        409: {"model": ErrorEnvelope, "description": "Exists with different content; use PUT"},
        413: {"model": ErrorEnvelope},
        415: {"model": ErrorEnvelope},
        422: {"model": ErrorEnvelope},
    },
    tags=["documents"],
    summary="Add a document",
)
async def create_document(
    request: Request,
    services: ServicesDep,
    file: Annotated[UploadFile, File(description="PDF, TXT or Markdown")],
    doc_id: Annotated[
        str | None,
        Form(description="Defaults to a slug of the filename", examples=["handbook"]),
    ] = None,
) -> JSONResponse:
    """Add a document, or report that it is already here unchanged.

    409 rather than a silent replace: this verb means "add", and quietly overwriting on a
    different payload is how a caller loses the version it thought it had.
    """
    data = await file.read()
    result = await services.ingestion.ingest(data, file.filename or "document", doc_id=doc_id)
    if result.status == "updated":
        return error(
            status.HTTP_409_CONFLICT,
            DocumentExistsError.code,
            f"{result.doc_id} already exists with different content; "
            f"PUT /documents/{result.doc_id} to replace it",
            request,
        )
    return JSONResponse(
        status_code=201 if result.status == "created" else 200,
        content=_ingest_body(result),
    )


@router.put(
    "/documents/{doc_id}",
    response_model=IngestResponse,
    responses={404: {"model": ErrorEnvelope}, **INVALID_QUERY},
    tags=["documents"],
    summary="Replace an active document",
)
async def replace_document(
    doc_id: str,
    services: ServicesDep,
    file: Annotated[UploadFile, File()],
) -> JSONResponse:
    """Replace an **active** document. 404 for unknown *or* deleted: a tombstone is not
    something to write to, and re-adding is a POST."""
    document = services.store.get_document(doc_id)
    if document is None or document["status"] != "active":
        raise DocumentNotFoundError(f"no active document with id {doc_id!r}")
    data = await file.read()
    result = await services.ingestion.ingest(data, file.filename or doc_id, doc_id=doc_id)
    return JSONResponse(status_code=200, content=_ingest_body(result))


@router.delete(
    "/documents/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorEnvelope}, **INVALID_QUERY},
    tags=["documents"],
    summary="Delete a document",
)
async def delete_document(doc_id: str, services: ServicesDep) -> Response:
    """Make a document unretrievable. 404 on a repeat, not a second 204."""
    await services.ingestion.delete(doc_id)
    # `Response`, not `JSONResponse`: a 204 has no body, and JSONResponse would write the
    # literal "null" into one.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/documents",
    response_model=DocumentList,
    responses=INVALID_QUERY,
    tags=["documents"],
    summary="List documents",
)
async def list_documents(
    services: ServicesDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    """Active documents only. A tombstone is an implementation detail, not a listing row."""
    doc_ids = services.store.documents(status="active")
    page = doc_ids[offset : offset + limit]
    return {
        "items": [_summary(services, doc_id) for doc_id in page],
        "total": len(doc_ids),
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentDetail,
    responses={404: {"model": ErrorEnvelope}, **INVALID_QUERY},
    tags=["documents"],
    summary="Document metadata and section outline",
    description=(
        "One document's current version and its section outline, read from SQLite -- the "
        "source of truth. Superseded versions are not listed; a deleted document is a 404."
    ),
)
async def get_document(doc_id: str, services: ServicesDep) -> dict[str, Any]:
    """`doc_id` defaults to a slug of the uploaded filename."""
    document = services.store.get_document(doc_id)
    if document is None or document["status"] != "active":
        raise DocumentNotFoundError(f"no active document with id {doc_id!r}")
    version = int(document["current_version"])
    chunks = services.store.chunks_for_version(doc_id, version)
    return {
        "doc_id": doc_id,
        "title": document["title"],
        "document": document["source_name"],
        "format": document["format"],
        "status": document["status"],
        "current_version": version,
        "last_version": int(document["last_version"]),
        "chunk_count": len(chunks),
        "language": document["language_hint"],
        "created_at": document["created_at"],
        "updated_at": document["updated_at"],
        "published_at": document["published_at"],
        "sections": _outline(chunks),
    }


# ---------------------------------------------------------------- query


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={422: {"model": ErrorEnvelope}},
    tags=["query"],
    summary="Answer a question from the corpus, or say the information is insufficient",
)
async def run_query(
    body: Annotated[QueryRequest, Body(openapi_examples=QUERY_EXAMPLES)],
    services: ServicesDep,
) -> dict[str, Any]:
    """Retrieve, gate, then answer -- or refuse. Always 200 when the service is healthy."""
    answer = await services.retrieval.answer(
        body.question,
        top_k=body.top_k,
        doc_ids=body.doc_ids,
        language=body.language,
        debug=body.debug,
    )
    return _answer_body(answer, debug=body.debug)


# ---------------------------------------------------------------- mapping


def _ingest_body(result: IngestResult) -> dict[str, Any]:
    return {
        "doc_id": result.doc_id,
        "version": result.version,
        "status": result.status,
        "chunks_added": result.chunks_added,
        "chunks_reused": result.chunks_reused,
        "chunks_removed": result.chunks_removed,
        "embed_requests": result.embed_requests,
        "duration_ms": result.duration_ms,
    }


def _summary(services: Services, doc_id: str) -> dict[str, Any]:
    document = services.store.get_document(doc_id) or {}
    version = int(document.get("current_version") or 0)
    return {
        "doc_id": doc_id,
        "title": document.get("title", doc_id),
        "document": document.get("source_name", ""),
        "format": document.get("format", ""),
        "current_version": version,
        "chunk_count": services.store.chunk_count(doc_id, version),
        "language": document.get("language_hint"),
        "updated_at": document.get("updated_at", ""),
    }


def _outline(chunks: list[dict[str, Any]]) -> list[Section]:
    """Distinct section breadcrumbs in document order, from the version's own chunks.

    Derived rather than stored: the chunks already carry it, and a second copy of the
    outline would need its own migration and could disagree with the text it describes.
    """
    seen: dict[str, Section] = {}
    for chunk in chunks:
        path = " > ".join(json.loads(chunk["section_path_json"]))
        if path in seen:
            continue
        lines = (
            [int(chunk["line_start"]), int(chunk["line_end"])]
            if chunk["line_start"] is not None and chunk["line_end"] is not None
            else None
        )
        seen[path] = Section(
            path=path,
            char_start=int(chunk["char_start"]),
            char_end=int(chunk["char_end"]),
            page=chunk["page_start"],
            lines=lines,
        )
    return list(seen.values())


def _answer_body(answer: Answer, *, debug: bool) -> dict[str, Any]:
    return {
        "status": answer.status,
        "answer": answer.answer,
        "segments": [
            {
                "text": segment.text,
                "citation_id": segment.citation_id,
                "chunk_char_start": segment.chunk_char_start,
                "chunk_char_end": segment.chunk_char_end,
            }
            for segment in answer.segments
        ],
        "evidence_score": answer.evidence_score,
        "citations": [citation.as_dict() for citation in answer.citations],
        "reason": answer.reason,
        "debug": answer.as_dict(debug=True)["debug"] if debug else None,
    }
