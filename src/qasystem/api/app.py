"""FastAPI app factory (plan.md §3, §7 P0).

P0 ships liveness only. The app is built once per process; SQLite and the local
Chroma directory are owned by that single process (``--workers 1``, plan.md L2).
"""

from __future__ import annotations

from fastapi import FastAPI

from qasystem import __version__
from qasystem.config import Settings, get_settings
from qasystem.logging_setup import configure_logging

DESCRIPTION = (
    "LLM-free extractive document QA. Retrieval finds evidence; answers are "
    "verbatim source slices with citations, or insufficient_information."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI app. Invalid configuration fails here, not per request."""
    resolved = settings or get_settings()
    configure_logging(resolved)

    app = FastAPI(
        title="Document QA",
        version=__version__,
        description=DESCRIPTION,
    )
    app.state.settings = resolved

    @app.get("/health", tags=["ops"], summary="Liveness")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
