# AGENTS.md
Project: LLM-free extractive document QA (FastAPI + SQLite/FTS5 + local persistent ChromaDB).
Source of truth: plan.md (read §0–§3 + your phase only).

Commands: `make check` (ruff, mypy, pytest) · `make run` · `make smoke` · `make eval` · `make calibrate`
Rules:
- Test first. Never edit/skip tests to get green. No real network unless RUN_LIVE=1.
- Never print/log/commit EMBEDDING_API_KEY. `.env` and `data/` are git-ignored.
- Chroma = chromadb.PersistentClient(path=CHROMA_PATH) ONLY. No HttpClient, no server, one worker.
- SQLite decides what is searchable (`eligible_chunks` view). Chroma/FTS hits are candidates, never evidence.
- No LLM calls. Answers are verbatim source slices with citations; otherwise `insufficient_information`.
- Verify third-party API behavior with a probe test; pin versions.
- After 3 failed attempts on one problem: stop and report.
- Log decisions with evidence in docs/DECISIONS.md.