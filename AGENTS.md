# AGENTS.md
Project: LLM-free extractive document QA (FastAPI + SQLite/FTS5 + local persistent ChromaDB).
Source of truth: `plan.md` (read §0–§3 + your phase only). Remaining phase: **P11**, planned in
`refactor-plan.md`. **There is no CI** — the Makefile is the whole gate, and only you run it.

## Commands

```bash
uv sync                                          # uv.lock is committed; deps are pinned exactly
make check                                       # ruff check + ruff format --check + mypy + pytest
make live                                        # RUN_LIVE=1 pytest tests/integration — real BGE-M3
make check-live                                  # the full definition of done: check, then live
uv run pytest tests/unit/test_gate.py::test_x -q # one test; add -k "name" to filter
RUN_LIVE=1 uv run pytest                         # 120 live tests — see the trap below
uv run pytest --cov=src/qasystem                 # coverage is measured, NOT gated by make check
make run                                         # uvicorn, --workers 1 is mandatory
make eval / make calibrate                       # own index under data/eval; never touches data/
scripts/smoke_test.sh                            # needs `make run` already serving
```

**`make live` is not the whole live suite.** It runs `pytest tests/integration` only, so the
7 tests in `tests/api/test_api_live.py` never run under it. Use `RUN_LIVE=1 uv run pytest`
when you touch the HTTP layer. Offline suite is 640 collected (598 pass / 42 skip).

## Rules

- Test first. Never edit or skip a test to get green.
- **Every commit needs `make check` *and* `make live`.** The fake embedder is a bag of hashed
  tokens: it cannot rank, cannot align languages, and its similarity scale is nothing like the
  real model's. The first live run found four defects that 480 offline tests could not (D45–D48),
  including a gate that could not authorise the system's own cross-lingual capability.
- Never print, log, or commit `EMBEDDING_API_KEY`. `.env` and `data/` are git-ignored.
- Chroma = `chromadb.PersistentClient(path=CHROMA_PATH)` ONLY. No HttpClient, no server, one
  worker — one process owns `data/` and the lock, so a second instance must fail fast.
- SQLite decides what is searchable (the `eligible_chunks` view). Chroma/FTS hits are candidates,
  never evidence.
- No LLM calls, ever — the provider has no chat endpoint. Answers are verbatim source slices
  with citations; otherwise `insufficient_information`. No silent model/version fallbacks.
- Verify third-party API behavior with a probe test; pin versions.
- After 3 failed attempts on one problem: stop and report.
- Log decisions with evidence in `docs/DECISIONS.md` (D1–D72, numbered, never renumbered).

## Configuration gotchas

- **The model id is configured, not auto-discovered.** `EMBEDDING_MODEL` must be set *and*
  appear in `GET /v1/models` or startup raises `EmbeddingAuthError` listing what is allowed.
  The **dimension is probed** with one real call; `EMBEDDING_DIMENSION` is optional and, if
  set, must equal the probe or startup refuses to mix dimensions (I9).
- **`EMBEDDING_PROVIDER=fake` only works when `APP_ENV` is `test` or `demo`.** `dev`/`prod`
  with the fake provider is a `ConfigError`, so you cannot smoke the app offline in `dev`.
- `SENTENCE_RERANK` in `.env.example` is **dead** — nothing reads it (D65). Don't wire it up
  or document it as a feature.
- Every tunable is a `Settings` field validated in one `model_validator`. No inline literals;
  provider caps (200k chars/request, 120 req/min) are enforced there, not in the client.

## Testing quirks

- `tests/conftest.py` monkeypatches `socket.connect` to raise unless `RUN_LIVE=1`, so a test
  that opens a socket **fails**, it does not skip. `_isolate_settings_env` strips every
  `Settings` field from the environment, so a test can never depend on your `.env`.
- `addopts = "--strict-markers"` — a new `pytest.mark.x` fails until you declare it in
  `pyproject.toml`. There is no `invariant` marker yet (P11 adds one).
- `ruff format --check` is in the gate: run `uv run ruff format src tests` before committing,
  or CI-by-hand fails on formatting alone.
- The Persian book under `tests/fixtures/docs/fa/` is committed on purpose (749 KB) — it is
  the only proof NFKC is required. `tests/unit/test_repo_consistency.py` fails if any fixture
  the suite reads is untracked, because a green tree hid a missing fixture for four phases.
- Live tests skip cleanly without a token rather than erroring.

## Known-stale claims (verified, do not "fix" the code to match)

- `plan.md` §7's layout lists `scripts/eval.sh`, which never existed.
- `docs/DECISIONS.md` contradicts itself about the eval report's path (D215/D1654 vs D1442).
  D74 settles it: the report is `docs/eval_report.md`, and a test now fails if a link breaks.
- Cross-lingual retrieval is **not** supported — D71 removed the branch that carried it. The
  README must say so; do not reintroduce an English-query-against-Persian-doc claim.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->