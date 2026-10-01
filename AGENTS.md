# AGENTS.md
Project: LLM-free extractive document QA (FastAPI + SQLite/FTS5 + local persistent ChromaDB).
Source of truth: plan.md (read §0–§3 + your phase only).

Commands: `make check` (ruff, mypy, pytest) · `make live` (same tests, real provider) · `make run` · `make smoke` · `make eval` · `make calibrate`
Rules:
- Test first. Never edit/skip tests to get green.
- **Every phase runs `make live`, not just `make check`.** The fake embedder is a bag of hashed
  tokens: it cannot rank, cannot align languages, and its similarity scale is nothing like the
  real model's. The first live run found four defects that 480 offline tests could not (D45-D48),
  including a gate that could not authorise the system's own cross-lingual capability.
- Never print/log/commit EMBEDDING_API_KEY. `.env` and `data/` are git-ignored.
- Chroma = chromadb.PersistentClient(path=CHROMA_PATH) ONLY. No HttpClient, no server, one worker.
- SQLite decides what is searchable (`eligible_chunks` view). Chroma/FTS hits are candidates, never evidence.
- No LLM calls. Answers are verbatim source slices with citations; otherwise `insufficient_information`.
- Verify third-party API behavior with a probe test; pin versions.
- After 3 failed attempts on one problem: stop and report.
- Log decisions with evidence in docs/DECISIONS.md.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
