# syntax=docker/dockerfile:1
#
# Production image for the LLM-free extractive QA service.
#
# Three constraints drive every line here, and each is a real invariant rather than a
# preference (docs/DECISIONS.md, plan.md §4.1):
#
# * **One worker, always.** One process owns `data/` and holds `FileLock(timeout=0)`. A second
#   worker does not scale the app, it makes the second one fail at startup. `--workers 1` is in
#   the CMD and must stay there.
# * **The token is never in the image.** It is read from the environment at runtime
#   (`SecretStr`, I8). `.dockerignore` keeps `.env` out of the build context so it cannot land
#   in a layer, and nothing here does `COPY . .`.
# * **The index is disposable.** `data/` is a volume, never baked in, so a container can be
#   replaced without replacing the corpus and `rebuild` can restore dense search with no API
#   calls (I10). Backup is "stop, copy data/".

ARG PYTHON_VERSION=3.13

# ------------------------------------------------------------------ build
FROM ghcr.io/astral-sh/uv:${PYTHON_VERSION} AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: this is only rebuilt when a pin changes, not on
# every source edit. `--locked` makes uv fail rather than silently re-resolve, so the image
# cannot drift from the committed uv.lock.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ------------------------------------------------------------------ runtime
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="Document QA" \
      org.opencontainers.image.description="LLM-free extractive document QA (SQLite + FTS5 + local persistent ChromaDB)"

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=prod \
    DATA_DIR=/data \
    SQLITE_PATH=/data/qasystem.db \
    CHROMA_PATH=/data/chroma \
    THRESHOLDS_PATH=/app/config/thresholds.json

# The venv was built against /app, so /app must exist at the same path in the runtime image.
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# `qasystem` itself, plus `config/thresholds.json`. The thresholds file is NOT inside the
# package -- `uv_build` does not ship it -- and a missing file does not crash: `load_thresholds`
# falls back to uncalibrated defaults (D42). That silent fallback is exactly why this line
# exists: without it the container would come up healthy and refuse far more questions than it
# should. THRESHOLDS_PATH is set explicitly rather than relying on the working directory.
COPY --from=builder /app/src /app/src
COPY config/ /app/config/

# Non-root, and `data/` created and owned by it up front so the volume inherits the ownership
# instead of the app failing to open its own database on first boot.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

VOLUME ["/data"]
EXPOSE 8000

# /ready is the honest probe: 503 while the model, the lock or the collection is unavailable,
# 200 when the service can actually answer. stdlib urllib, because slim has no curl and adding
# one to a healthcheck is a package nobody asked for.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"]

# `--workers 1` is L2, not a default. See the header.
CMD ["uvicorn", "qasystem.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]