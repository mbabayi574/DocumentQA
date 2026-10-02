"""FastAPI app: lifespan, one error envelope, and a request-id middleware (plan.md P8).

Three things here are load-bearing and belong nowhere else:

* **The lifespan owns the process.** It builds the service graph -- which takes the
  data-directory lock (L2) and probes the embedding model once (L9) -- and releases both on
  shutdown. A failure inside it stops the process from starting rather than surfacing as a
  500 on the first request, which is the difference between a clear message and a mystery.
* **One error shape for everything.** Domain errors carry their own ``code`` and HTTP status;
  validation errors become ``VALIDATION_ERROR``; anything unexpected becomes
  ``INTERNAL_ERROR`` with no detail. No route can invent a third shape, and no stack trace
  and no provider message ever reaches a client.
* **The middleware logs identifiers, counts, timings and error classes only.** Never a body,
  never a header value, never a query string -- a URL can carry a question, and a question
  can carry a secret.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from qasystem import __version__
from qasystem.api.deps import Services, build_services
from qasystem.api.routes import router
from qasystem.config import Settings, get_settings
from qasystem.errors import QASystemError
from qasystem.logging_setup import configure_logging, scrub, secrets_of

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

DESCRIPTION = (
    "LLM-free extractive document QA. Retrieval finds evidence; answers are "
    "verbatim source slices with citations, or insufficient_information."
)


def create_app(settings: Settings | None = None, *, services: Services | None = None) -> FastAPI:
    """Build the ASGI app.

    ``services`` is injectable for tests and the CLI; without it the lifespan builds the
    graph itself, which is what ``uvicorn --factory`` does.
    """
    resolved = settings or (services.settings if services else get_settings())
    configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        graph = services or await build_services(resolved)
        app.state.services = graph
        try:
            yield
        finally:
            # Only close a graph we built. An injected one is the caller's to own, and
            # closing it here would leave a test with a dead store it still holds.
            if services is None:
                await graph.aclose()

    app = FastAPI(
        title="Document QA", version=__version__, description=DESCRIPTION, lifespan=lifespan
    )
    app.state.settings = resolved
    app.include_router(router)
    _install_error_handlers(app, resolved)
    _install_request_id(app)
    return app


def _envelope(code: str, message: str, request_id: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def _install_error_handlers(app: FastAPI, settings: Settings) -> None:
    """Every failure leaves through one of these, so the shape is a property of the app."""

    @app.exception_handler(QASystemError)
    async def domain_error(request: Request, exc: QASystemError) -> JSONResponse:
        # A typed domain error's message is written for a caller, but scrub anyway: one of
        # them can be built from a provider response, and I8 has no exceptions.
        return JSONResponse(
            status_code=exc.http_status,
            content=_envelope(
                exc.code, scrub(secrets_of(settings), str(exc)), request.state.request_id
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "VALIDATION_ERROR",
                _summarise_validation(exc),
                request.state.request_id,
            ),
        )

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The class name is safe to log and useless to a caller; the message may contain a
        # path, a query fragment or a provider body, so it stays in the log.
        logger.warning(
            "unhandled error: %s", type(exc).__name__, exc_info=False, extra={"event": "error"}
        )
        return JSONResponse(
            status_code=500,
            content=_envelope("INTERNAL_ERROR", "internal error", request.state.request_id),
        )


def _install_request_id(app: FastAPI) -> None:
    """Stamp every response, and log one line per request with no payload in it."""

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request.state.request_id = incoming or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.info(
                "%s %s -> 500 in %.1fms id=%s",
                request.method,
                request.url.path,
                (time.perf_counter() - started) * 1000,
                request.state.request_id,
            )
            raise
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        logger.info(
            "%s %s -> %d in %.1fms id=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request.state.request_id,
        )
        return response


def _summarise_validation(exc: RequestValidationError) -> str:
    """Field names and reasons only. The offending value could be a document's contents."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ())) or "body"
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return "; ".join(parts)[:500]
