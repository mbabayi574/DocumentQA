# Document-Based QA System (LLM-free Extractive RAG)

> **Audience:** AI coding agents (and the human reviewing them) implementing `jobTask.md`.
> **Version:** v2 — restructured for agent execution; **ChromaDB now runs in local, persistent (on-disk) mode**.
>
> **Core decision:** The only model service available is **embedding-only** (`GET /v1/models`, `POST /v1/embeddings`). There is no chat endpoint, so the system is deliberately **LLM-free and extractive**: retrieval finds evidence, and the answerer returns **verbatim source text with citations**. Nothing in the system may paraphrase, summarize, translate, or complete an answer.

---

## 0. Agent Operating Protocol — read first, every session

1. **Load minimal context.** Read §0–§3 plus only the phase you are executing. Do not re-read the whole file.
2. **One phase per branch/session.** Work test-first: write the phase's *Tests first* list (red), implement (green), refactor.
3. **Gate before commit.** `make check` (= `ruff check` + `ruff format --check` + `mypy src` + `pytest -q`) must be green, and the phase's **Gate** items must pass.
4. **Never weaken tests to get green.** If a test looks wrong, log it in `docs/DECISIONS.md` and flag it to the human.
5. **Verify third-party behavior, don't recall it.** After installing `chromadb`, `pymupdf`, `markdown-it-py`, etc., pin versions in `pyproject.toml` and write a tiny probe test for every behavior you depend on (e.g. Chroma cosine distance, FTS5 availability, persistence across client restarts).
6. **No real network in tests.** Live calls only when `RUN_LIVE=1`. Never print the token.
7. **Secrets:** read `EMBEDDING_API_KEY` from env only. Never commit, log, echo, snapshot, or return it.
8. **Scope guard — do NOT add:** OCR, auth, GUI, message queues, any LLM/chat call, other vector DBs, Chroma HTTP/server mode, silent fallbacks to other models or versions.
9. **Stop-and-report rule.** After 3 failed attempts on the same failing test, stop, summarize the hypothesis and evidence, and ask. Do not hack around it.
10. **Commit small,** one logical change each: `feat(ingestion): publish version atomically` (tests included).
11. **Record decisions with evidence** in `docs/DECISIONS.md` (model choice, thresholds, chunk size). Reviewers weigh reasoning over tool count.
12. **Keep chat output small.** Never dump embeddings, full documents, or whole files into the conversation.

### Execution map (parallel lanes)

After **P0** defines the contracts, lanes A–D are independent and can be assigned to separate agents/branches:

```text
P0 Scaffold + contracts ──┬─ Lane A: P1 Text layer ─► P3 Chunking ──┐
                          ├─ Lane B: P2 Parsing ───────────────────┤
                          ├─ Lane C: P4 Embeddings client + cache ─┼─► P6 Ingestion ─► P7 Retrieval+Answer ─► P8 API ─► P9 Eval ─► P10 Harden+README
                          └─ Lane D: P5 Storage (SQLite/FTS/Chroma)┘
```

| Phase | Size | Depends on | Risk |
|---|---|---|---|
| P0 Scaffold + contracts | S | — | low |
| P1 Text layer | S | P0 | low |
| P2 Parsing | M | P0 | medium (PDF) |
| P3 Chunking | M | P1, P2 | medium |
| P4 Embeddings client + cache | M | P0 | medium |
| P5 Storage | M | P0 | medium |
| **P6 Ingestion / change management** | **L** | P3, P4, P5 | **highest** |
| P7 Retrieval + gate + answering | L | P6 | high |
| P8 API | M | P7 | low |
| P9 Evaluation + model selection + calibration | M | P8 | medium |
| P10 Hardening + README | S | P9 | low |

---

## 1. Requirement → Design → Proof traceability

| `jobTask.md` requirement | Design answer | Proof (test / artifact) |
|---|---|---|
| PDF, TEXT, Markdown processing | Parser per format → common `ParsedDocument` with char/page/line offsets | P2 fixture tests |
| Add / edit / delete documents | Versioned documents; publication = one SQLite pointer flip; Chroma is derived | P6 invariant + failure-injection tests |
| Outdated/deleted content never used | Single `eligible_chunks` view gates **every** retrieval path | I1–I3 tests, stale-leakage metric = 0 |
| Relevant retrieval | Hybrid: dense (Chroma) + lexical (FTS5) → weighted RRF | P9 Recall@k / MRR vs baselines |
| Evidence-based, traceable answers | Extractive answerer; citations carry doc, version, section, page/line, char offsets | I6/I7 substring tests |
| Insufficient information handling | Evidence gate **before** answer selection; fixed system message, zero citations | P9 false-answer rate |
| API access without GUI | FastAPI + OpenAPI; CLI for maintenance | P8 API tests |
| Choose & justify an embedding model | Measured comparison of reachable candidates; decision logged | P9 report + `docs/DECISIONS.md` |
| Code quality, separation of concerns, errors, tests | Ports/adapters, typed errors, invariant-driven tests | `make check`, coverage targets |

---

## 2. Ground Truth, Constraints, Assumptions

| ID | Constraint | Consequence |
|---|---|---|
| C1 | Formats: PDF, TXT, Markdown. | One parser per format; shared `ParsedDocument`. |
| C2 | Edits/deletes must never leave stale evidence queryable. | Version-publication is the core consistency problem (P6). |
| C3 | Model service exposes only `GET /v1/models` and `POST /v1/embeddings`. | No generation. Extractive answers only. |
| C4 | Candidate models (`jobTask.md`): BGE-M3, Embedding-3-Small, Embedding-3-Large, Gemini-Embedding-001. | **Do not hard-code IDs or dimensions.** Discover exact IDs from `/v1/models` (casing may differ, e.g. `Bge-m3`) and probe the vector dimension with one real call. Skip any candidate that is unreachable. Never silently switch models. |
| C5 | Limits: **120 requests/min**, **200,000 Characters/request** (per `jobTask.md`). | Client-side limiter ≤ 100 req/min. Batch on **both** a character budget (default 190,000 — stricter than the token limit in practice, so it satisfies either reading) **and** an item cap. Retry 429/5xx/timeouts with bounded backoff. |
| C6 | Bearer token is confidential. | `SecretStr`, env-only, redaction filter in logging. |
| C7 | **ChromaDB runs locally in persistent on-disk mode** (`chromadb.PersistentClient`), in-process, at `CHROMA_PATH` (default `./data/chroma`). No Chroma server, no Podman/Compose, no Kafka/Qdrant. | See **§3.2 Local Chroma rules**. |
| C8 | Evaluation weighs retrieval quality, substantiation, change management, readability, error handling, tests. | Correctness and clarity over architectural breadth. |

**Assumption A1 — Persian support.** `jobTask.md` does not mention languages. Persian/English support is assumed from the provider's locale. Implement English first; add Persian normalization/fixtures as a bounded second-class workstream (~15% of effort). State this assumption in the README.

### Non-negotiable invariants (release blockers, each encoded as a test)

| ID | Invariant |
|---|---|
| I1 | **Active-version:** every evidence item comes from a chunk whose `doc_version == documents.current_version` of an `active` document. |
| I2 | **Deletion:** after the delete transaction commits, the document is unretrievable via lexical *and* dense paths even if Chroma cleanup never ran. |
| I3 | **Atomic publication:** a failure before the publish transaction leaves the previous version fully queryable and unchanged. |
| I4 | **Idempotency:** re-uploading identical bytes (or bytes producing identical parsed text) → `unchanged`, **zero** embedding requests, no new version. |
| I5 | **Reuse:** unchanged chunks cause zero embedding requests (cache keyed by the exact embedded input). |
| I6 | **Traceability:** `source_text[char_start:char_end] == chunk.text`, and every answer segment is an exact substring of its cited chunk. |
| I7 | **No synthesis:** answer text = selected source slices in source order (+ `[n]` markers rendered outside the slices). |
| I8 | **Secrets:** the token never appears in logs, errors, snapshots, fixtures, or committed files. |
| I9 | **Model isolation:** one Chroma collection per `(model_id, dimension)`; vectors of different models/dimensions are never mixed. |
| I10 | **Disposable index:** deleting `data/chroma/` and running `rebuild` restores dense search from SQLite with **zero** embedding API calls. |

---

## 3. Architecture

```text
            +-------------------------------+
            |            FastAPI            |   (single process, 1 worker)
            | upload / update / delete /    |
            | query / health               |
            +---------------+---------------+
                            |
                  application services
                            |
      +---------------------+----------------------+
      |                     |                      |
   Parsing             Ingestion               Retrieval ──► Gate ──► Extractive answer
 (md/txt/pdf)     (diff → stage → publish)   (dense + lexical → RRF)
      |                     |                      |
      +---------------------+----------+-----------+
                                       |
                  +--------------------+--------------------+
                  |                                         |
         SQLite  (data/qasystem.db)               Chroma PersistentClient (data/chroma/)
      source of truth · versions · chunks         derived dense index · disposable
      FTS5 lexical · embedding cache

     External (network): embedding API only — POST /v1/embeddings, GET /v1/models
```

### 3.1 Key design decisions

- **SQLite is the single source of truth:** documents, versions, chunk text/offsets, FTS5, embedding cache, audit log. Publication state lives **only** here.
- **Chroma is a derived, rebuildable dense index** that stores vectors + small diagnostic metadata only (no document text). Because SQLite's `embedding_cache` keeps every vector, `rebuild` needs no API calls (I10).
- **Publish = one SQLite transaction** that flips `documents.current_version`. There is no second "is_active" flag to keep in sync; eligibility is *derived* through a view (§5).
- **Versions are monotonic per `doc_id`, even across delete/re-add.** Vector IDs are `"{doc_id}:v{version}:{ordinal}"`, so a stale vector can never collide with a fresh one.
- **Publish, then clean up.** Embed and stage everything first; publish atomically; delete superseded vectors/rows best-effort afterwards. An incomplete update is ignorable; a published update is complete.
- No message queue: the API shape does not need one.

### 3.2 Local Chroma rules (persistent, on-disk)

| # | Rule |
|---|---|
| L1 | Create the client with `chromadb.PersistentClient(path=settings.chroma_path, settings=Settings(anonymized_telemetry=False))`. **Never** use `HttpClient`, `EphemeralClient`, or an in-memory fallback outside tests. |
| L2 | **Single-owner process.** Run `uvicorn --workers 1`. At startup acquire an exclusive file lock (`filelock`, `timeout=0`) on `data/.qasystem.lock`; if it is held, exit with a clear error. Multiple processes must never open the same persist directory. |
| L3 | Chroma calls are blocking: wrap them in `asyncio.to_thread`. Serialize writes (Chroma + SQLite) behind one application-level write lock. |
| L4 | Collection name is deterministic and Chroma-safe: `chunks__{slug(model_id)}__d{dimension}` (`[a-z0-9_-]`, < 63 chars). Store `model_id`, `dimension` in collection metadata and **assert** them on open (I9). |
| L5 | Use cosine space via the mechanism the **pinned** Chroma version supports (`metadata={"hnsw:space": "cosine"}` or `configuration={"hnsw": {"space": "cosine"}}`); prove it with a known-vector test. Similarity = `1 − distance`. |
| L6 | Store **no document text** in Chroma (pass embeddings + metadata only). Metadata: `doc_id`, `doc_version`, `ordinal`. Chroma metadata is diagnostic; SQLite filtering decides eligibility. |
| L7 | Clamp `n_results` to `collection.count()` and skip the query when the count is 0 (Chroma errors/warns otherwise). Over-fetch (`CANDIDATES_N × OVERFETCH`, default 2) so stale vectors that get filtered out don't starve the candidate list. |
| L8 | `data/` is git-ignored. **Backup = stop the app, copy `data/`.** The app never mutates Chroma's internal files directly. |
| L9 | Readiness checks: data dir writable, lock held, `client.heartbeat()`, collection opens, dimension matches config. A failure is reported explicitly — never silently recreate a different store. |
| L10 | Tests use a real `PersistentClient` on `tmp_path` (fast, no external service). A **restart test** must: write → drop client → reopen on the same path → query → still found. |

---

## 4. Contracts (define in P0, keep stable)

Put these in `src/qasystem/domain/`. They are the seams that let lanes A–D run in parallel. The domain layer must not import FastAPI, Chroma, httpx, or sqlite3 (enforce with an AST-based test).

```python
# domain/models.py
@dataclass(frozen=True)
class Section:                     # produced by parsers
    char_start: int; char_end: int             # slice of ParsedDocument.text
    section_path: tuple[str, ...]              # ("Guide", "Install", "Linux")
    page_start: int | None; page_end: int | None
    line_start: int | None; line_end: int | None

@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str                                  # "source text": what all offsets refer to
    sections: tuple[Section, ...]
    format: Literal["pdf", "txt", "md"]

@dataclass(frozen=True)
class Chunk:                       # produced by chunker; text == source_text[char_start:char_end]
    ordinal: int
    char_start: int; char_end: int
    text: str
    section_path: tuple[str, ...]
    page_start: int | None; page_end: int | None
    line_start: int | None; line_end: int | None
    chunk_hash: str                            # sha256(canonical(section_path) + normalized text)
    language: Literal["fa", "en", "mixed"]

@dataclass(frozen=True)
class VectorItem:  id: str; vector: Sequence[float]; metadata: Mapping[str, str | int]
@dataclass(frozen=True)
class VectorHit:   id: str; similarity: float
```

```python
# domain/ports.py
class Embedder(Protocol):
    model_id: str
    dimension: int
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...   # order-preserving

class VectorStore(Protocol):
    def ensure_collection(self, model_id: str, dimension: int) -> None: ...
    def upsert(self, items: Sequence[VectorItem]) -> None: ...
    def query(self, vector: Sequence[float], n: int) -> list[VectorHit]: ...
    def get_existing_ids(self, ids: Sequence[str]) -> set[str]: ...
    def list_ids(self) -> Iterator[str]: ...            # paged; for reconcile
    def delete_ids(self, ids: Sequence[str]) -> None: ...
    def count(self) -> int: ...
    def ping(self) -> None: ...                         # raises VectorStoreError

class DocumentParser(Protocol):
    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...
```

Application services (not ports): `IngestionService`, `RetrievalService`, `AnswerService`. Only add a new Protocol when it creates a real testing seam.

**Typed errors** (`errors.py`), each with a stable `code` and HTTP mapping:

| Error | Code | HTTP |
|---|---|---|
| `UnsupportedFormatError` | `UNSUPPORTED_FORMAT` | 415 |
| `FileTooLargeError` | `FILE_TOO_LARGE` | 413 |
| `EmptyDocumentError` / `ParseError` / `NoTextLayerError` | `EMPTY_DOCUMENT` / `PARSE_ERROR` / `NO_TEXT_LAYER` | 422 |
| `DocumentNotFoundError` | `DOCUMENT_NOT_FOUND` | 404 |
| `DocumentExistsError` (POST with existing active `doc_id`, different content) | `DOCUMENT_EXISTS` | 409 |
| `EmbeddingUnavailableError` (transient exhausted) / `EmbeddingAuthError` | `EMBEDDING_UNAVAILABLE` / `EMBEDDING_AUTH` | 503 / 502 |
| `VectorStoreError` (local Chroma failure) | `VECTOR_STORE_UNAVAILABLE` | 503 |
| `InsufficientEvidence` is **not** an error — it is a normal `200` response | — | 200 |
| anything else | `INTERNAL_ERROR` | 500 (no stack trace/secrets) |

---

## 5. Data Model (SQLite)

Pragmas: `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout`. Use short-lived connections (cheap in SQLite) and one write lock.

```sql
documents(
  doc_id TEXT PRIMARY KEY, title TEXT, source_name TEXT, format TEXT,
  status TEXT CHECK(status IN ('active','deleted')),
  current_version INTEGER NULL,           -- NULL while deleted
  last_version INTEGER NOT NULL,          -- monotonic allocator; never decreases
  content_hash TEXT, parsed_hash TEXT,    -- sha256(raw bytes) / sha256(parsed text)
  language_hint TEXT NULL, created_at TEXT, updated_at TEXT, deleted_at TEXT NULL)

document_versions(
  doc_id, version, content_hash, source_text TEXT,        -- parsed text, for I6 verification
  state TEXT CHECK(state IN ('staging','published','superseded','failed')),
  created_at, published_at, PRIMARY KEY(doc_id, version))

chunks(
  chunk_id TEXT PRIMARY KEY,              -- "{doc_id}:v{version}:{ordinal}"  == Chroma vector id
  doc_id, doc_version, ordinal, text TEXT NOT NULL,
  char_start INT, char_end INT,
  chunk_hash TEXT, embed_input_hash TEXT, -- hash of the exact string that was embedded
  section_path_json TEXT, page_start, page_end, line_start, line_end, language,
  UNIQUE(doc_id, doc_version, ordinal))

chunks_fts  -- FTS5(chunk_id UNINDEXED, tokens); tokens = OUR tokenizer's output, space-joined
            -- rows for staging AND published versions; eligibility is applied by join

embedding_cache(
  input_hash TEXT, model_id TEXT, dim INT, vector BLOB /* float32 */, created_at,
  PRIMARY KEY(input_hash, model_id))
  -- key = sha256 of the exact embedded string, so ANY change to the breadcrumb/template
  -- invalidates correctly; no separate "input version" column is needed.

ingest_log(id, doc_id, action, prev_version, new_version, prev_hash, new_hash,
           chunks_added, chunks_reused, chunks_removed, embed_requests,
           duration_ms, status, error_code, request_id, ts)   -- never bodies or secrets

-- THE choke point. Every retrieval path reads chunks ONLY through this view (I1, I2).
CREATE VIEW eligible_chunks AS
  SELECT c.* FROM chunks c
  JOIN documents d ON d.doc_id = c.doc_id
                  AND d.status = 'active'
                  AND d.current_version = c.doc_version;
```

Lexical search: store the tokenizer's output (incl. ZWNJ split components) in FTS5 so SQLite's tokenizer never has to understand Persian. Query with `bm25()`, joined to `eligible_chunks`. User text is tokenized and each token is emitted as a quoted FTS term joined by `OR` — **never** passed as raw FTS5 syntax. Startup check: FTS5 is available in this Python's SQLite build.

---

## 6. Configuration (`config.py`, validated at startup)

```dotenv
APP_ENV=dev                          # dev | test | demo | prod
SENTENCE_RERANK=false                # optional P7/P9 experiment; adds embedding requests per query

# --- embedding service ---
EMBEDDING_BASE_URL=https://models-interview.arvancloudai.ir/v1
EMBEDDING_API_KEY=<secret, required when provider=remote>
EMBEDDING_MODEL=<exact id from GET /v1/models>      # default candidate: BGE-M3 family
EMBEDDING_PROVIDER=remote            # remote | fake  (fake only when APP_ENV in {test, demo}; shown in /ready)
EMBEDDING_DIMENSION=                 # optional; if set, must equal the probed dimension

# --- storage (all local) ---
DATA_DIR=./data
SQLITE_PATH=./data/qasystem.db
CHROMA_PATH=./data/chroma            # PersistentClient path. No host/port/tenant settings exist.
MAX_UPLOAD_MB=20

# --- chunking ---
CHUNK_TARGET_TOKENS=350
CHUNK_HARD_MAX_TOKENS=700
CHUNK_OVERLAP_RATIO=0.15

# --- embedding client ---
MAX_CHARS_PER_REQUEST=160000
MAX_ITEMS_PER_BATCH=32
RATE_LIMIT_PER_MIN=100               # stay under the 120/min provider limit
REQUEST_TIMEOUT_S=30
MAX_RETRIES=5

# --- retrieval ---
CANDIDATES_N=30
OVERFETCH=2
TOP_K=5
RRF_K=60
DENSE_WEIGHT=0.7
LEXICAL_WEIGHT=0.3

# --- gate + answer (loaded from a versioned thresholds file produced by `make calibrate`) ---
THRESHOLDS_PATH=./config/thresholds.json
MAX_ANSWER_SENTENCES=5
```

`.env` is git-ignored; commit only `.env.example` with placeholders. Every tunable threshold lives in config, never inline.

---

## 7. Phases

Each phase lists: **Build**, **Tests first**, **Gate**. Do not start a dependent phase until the Gate passes.

### Repository layout (target)

```text
rag-qa/
├── AGENTS.md  README.md  plan.md  Makefile  pyproject.toml  .env.example  .gitignore
├── config/thresholds.json            # produced by `make calibrate`, versioned
├── docs/ DECISIONS.md  eval_report.md
├── src/qasystem/
│   ├── config.py  errors.py  logging_setup.py  cli.py
│   ├── domain/        models.py  ports.py
│   ├── text/          normalize.py  tokenize.py  language.py
│   ├── parsing/       base.py  markdown.py  text.py  pdf.py  registry.py
│   ├── chunking/      chunker.py
│   ├── embeddings/    client.py  rate_limit.py  caching.py  fake.py
│   ├── storage/       schema.sql  sqlite_store.py  lexical.py  chroma_store.py
│   ├── ingestion/     diff.py  service.py  reconcile.py
│   ├── retrieval/     fusion.py  gate.py  service.py
│   ├── answering/     sentences.py  extractive.py
│   └── api/           app.py  routes.py  schemas.py  deps.py
├── tests/ unit/ integration/ api/ eval/{dataset.jsonl,runner.py} fixtures/docs/{en,fa}/
├── data/                               # git-ignored: qasystem.db, chroma/, .qasystem.lock
└── scripts/ smoke_test.sh  eval.sh
```

Dependency direction: `api → services → domain/ports ← adapters (parsing, storage, embeddings)`.

---

### P0 — Scaffold, configuration, contracts  `[S]`

**Build**
- `uv` project, Python ≥ 3.11. Dependencies: `fastapi`, `uvicorn`, `python-multipart`, `pydantic>=2`, `pydantic-settings`, `httpx`, `pymupdf`, `markdown-it-py`, `chromadb` (**pinned**), `numpy`, `filelock`; dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `respx`, `hypothesis`, `ruff`, `mypy`.
- `Makefile` targets: `check`, `test`, `run`, `smoke`, `eval`, `calibrate`, `rebuild`.
- `.env.example`, `.gitignore` (`.env`, `data/`, caches), `AGENTS.md` (Appendix B).
- `config.py` (`SecretStr` for the key; validate paths, limits, provider rules), `errors.py` (§4), `logging_setup.py` (secret-redaction filter; log ids/counts/timings only).
- `domain/models.py`, `domain/ports.py` exactly as §4. `embeddings/fake.py`: deterministic, L2-normalized, hash-seeded vectors of configurable dimension.
- App factory with `GET /health` (liveness only).

**Tests first:** missing key → controlled `ConfigError`; key absent from `repr()`/logs; `FakeEmbedder` deterministic; domain package imports no fastapi/chromadb/httpx/sqlite3 (AST test); `fake` provider rejected when `APP_ENV=prod`.

**Gate:** `make check` green; app boots with `EMBEDDING_PROVIDER=fake APP_ENV=test`; no network used by any test.

---

### P1 — Text layer: normalization, tokenization, language  `[S]` · Lane A

**Build**
- `normalize_for_index(s)`: NFKC; Arabic `ي/ك → ی/ک`; Arabic-Indic/Persian digits → ASCII; strip tatweel and diacritics; collapse whitespace; **keep** ZWNJ (U+200C) but normalize it (no blind deletion); drop other control/zero-width chars. **Stored and cited text is never normalized** — only index/query strings.
- `tokenize(s)`: lowercase Latin; keep ZWNJ compounds **and** emit split components for recall; preserve identifiers (`ERR-404`, `v2.3.1`); no stemming unless P9 shows a measurable gain. Small built-in stopword sets (en/fa) used only for coverage/sentence scoring, not for FTS.
- `detect_language(s) -> fa|en|mixed` (script-ratio heuristic).

**Tests first:** Persian variants match; ZWNJ present/absent queries match the same fixture; idempotence (`f(f(x)) == f(x)`, Hypothesis); mixed fa/en and numeric identifiers; English unchanged apart from case/whitespace.

**Gate:** all green; functions are pure (no I/O).

---

### P2 — Parsing: Markdown, TXT, PDF  `[M]` · Lane B

**Build** — every parser returns a `ParsedDocument` whose offsets index its own `text`.
- **Markdown:** use `markdown-it-py` (token line maps) rather than regexes; ATX + Setext headings; ignore `#` inside fenced code; breadcrumb `Guide > Install > Linux`; text before the first heading takes the document title as root; convert line spans to char offsets.
- **TXT:** decode UTF-8-SIG/UTF-8; allow a Windows-1256 fallback only if a fixture justifies it, otherwise a typed decode error; split on blank lines; keep line spans. Normalize only `\r\n → \n`.
- **PDF (PyMuPDF):** page-by-page, `get_text("blocks", sort=True)`; exact 1-based page numbers recorded; headings only when font-size heuristics are reliable, else page-based sections; **never reverse RTL strings** unless a fixture proves the parser output needs it. Detect encrypted/corrupt PDFs and PDFs with no usable text layer → `NoTextLayerError` (there is no OCR; say so).
- **Registry:** choose by extension + PDF magic bytes (`%PDF-`); enforce `MAX_UPLOAD_MB` **before** parsing; unsupported → `UnsupportedFormatError`; empty/whitespace-only → `EmptyDocumentError`.

**Tests first:** English + Persian fixtures for MD/TXT; PDF fixtures committed as files (do not generate Persian PDFs in tests; if a reliable Persian PDF can't be produced, cover Persian via MD/TXT and document the gap); breadcrumbs, line spans, page numbers correct; corrupt PDF, scanned PDF, empty file, oversize file, wrong magic bytes.

**Gate:** fixtures pass; `text[s.char_start:s.char_end]` is non-empty for every section.

---

### P3 — Deterministic chunking  `[M]` · Lane A (after P1, P2)

**Build**
- Heading-aware: never merge unrelated top-level sections; merge small *adjacent sibling* sections only if the result stays ≤ target; split oversized sections with sentence-aware windows.
- Target ≈ `CHUNK_TARGET_TOKENS` with ≈ `CHUNK_OVERLAP_RATIO` overlap and a hard cap `CHUNK_HARD_MAX_TOKENS`. Use a documented heuristic (chars-per-token by script), not a fake exact tokenizer.
- Sentence boundaries: `. ! ? ؟ ۔`, avoiding decimals/abbreviations (`3.14`, `e.g.`, `Dr.`).
- **Every chunk is a contiguous slice** of `ParsedDocument.text` (overlap = earlier start offset). This makes I6 mechanically checkable.
- `chunk_hash = sha256(canonical(section_path) + normalize_for_index(text))`.
- Embedding input (built in the ingestion layer, not stored in Chroma): `"{title} > {section breadcrumb}\n\n{raw chunk text}"`; citations always use the raw slice.

**Tests first:** determinism (byte-identical output on re-run); every source paragraph covered by ≥ 1 chunk; no chunk over the hard cap; slice identity `text == source[char_start:char_end]`; local paragraph edit preserves ≥ 80% of unaffected `chunk_hash`es (Hypothesis + fixtures).

**Gate:** all green.

---

### P4 — Embedding client, batching, cache  `[M]` · Lane C

**Build**
- Async `httpx` client: `GET /models`, `POST /embeddings` with `{"model": ..., "input": [...]}`.
- **Discovery at startup:** configured model must exactly match an ID in `/models` (error lists available IDs, never the token). **Probe** dimension with one embedding call; if `EMBEDDING_DIMENSION` is set it must match.
- Reassemble vectors **by `index`**, not response order; validate every vector's dimension.
- Batch by character budget **and** item cap; token-bucket limiter ≤ `RATE_LIMIT_PER_MIN`.
- Retry only 429 / 5xx / connection resets / timeouts: honor `Retry-After`, else capped exponential backoff + jitter. 401/403 fail fast (`EmbeddingAuthError`). Exhausted retries → `EmbeddingUnavailableError`.
- `CachingEmbedder(inner, cache_repo)`: dedupe inputs within a call, look up `(sha256(input), model_id)`, embed only misses, write back. Small in-memory LRU for query embeddings.

**Tests first (respx, no real network):** order reassembly; batches never exceed limits; 429 + `Retry-After`; persistent 5xx → typed error; 401/403 no retry; dimension mismatch; model ID not in `/models`; duplicates in one call cause one embed; cache hit causes zero requests.
**Optional live test:** only with `RUN_LIVE=1`; never prints the token.

**Gate:** all green; limiter test proves ≤ configured rate under a burst (use a fake clock).

---

### P5 — Storage layer (SQLite + FTS5 + local Chroma)  `[M]` · Lane D

**Build**
- `schema.sql` per §5, idempotent creation, `PRAGMA user_version` for migrations; repository methods with one obvious transaction boundary per operation.
- `lexical.py`: token-quoted `OR` queries, `bm25()` rank, normalized lexical score, **always joined to `eligible_chunks`**.
- `chroma_store.py`: `VectorStore` over `PersistentClient` per §3.2 (L1–L10): deterministic collection name, cosine space, `similarity = 1 − distance`, `n` clamping, paged `list_ids`, `ping`.
- `ingestion/reconcile.py` primitives (used in P6/P10): `plan_reconcile()` compares SQLite's expected vector IDs with `list_ids()`; `rebuild()` re-upserts from `embedding_cache` (no API calls).

**Tests first:** schema idempotent; transaction rollback; `eligible_chunks` excludes staged/superseded/deleted rows; FTS with hostile input (`" OR * ( ) NEAR`); Chroma cosine conversion on known vectors; **restart persistence test** (L10); collection/dimension mismatch is rejected (I9); `n` clamp on tiny collections; second process cannot take the lock (L2).

**Gate:** storage tests run on `tmp_path` only; Chroma adapter is swappable for a fake in service tests.

---

### P6 — Ingestion and change management  `[L]` · **highest risk — finish before tuning retrieval**

**Pipeline**

```text
validate → parse → chunk → content/parsed hash → unchanged? ──yes──► 200 "unchanged" (0 embed calls)
                                   │no
                      diff vs currently published chunks (hash → queue of prior chunks)
                                   ▼
            resolve cached embeddings → embed ONLY missing inputs
                                   ▼
   STAGE (SQLite txn): new version row state='staging' + chunk rows + FTS rows
   STAGE (Chroma): upsert vectors under "{doc_id}:v{N}:{ordinal}"
   VERIFY: get_existing_ids(expected) == expected
                                   ▼
   PUBLISH (ONE SQLite txn): version→published, previous→superseded,
           documents.current_version=N, status='active', hashes updated
                                   ▼
   CLEANUP (best-effort, separate step): delete superseded chunk/FTS rows + Chroma ids
```

**Rules**
1. **Never** touch the published version before every embedding is resolved and staging is verified.
2. Version number is allocated from `documents.last_version + 1` inside a transaction; never reused.
3. Diff by `chunk_hash` using **multiset** matching (hash → queue) so duplicates keep multiplicity; reordering alone triggers zero embed calls. "Reused" means the **embedding** is reused (cache hit) and re-upserted under the new versioned ID — local Chroma upserts are free.
4. Any failure before publish → version marked `failed`, staged rows/vectors removed best-effort, previous version untouched (I3). Stragglers are swept by `reconcile`.
5. **Delete:** one transaction sets `status='deleted'`, `current_version=NULL`; from that commit on the document is unretrievable (I2). Then purge chunks/FTS/vectors best-effort. Keep the `documents` tombstone so `last_version` stays monotonic.
6. **Re-add** after delete creates a fresh version `last_version+1`.
7. If staging into Chroma fails, **abort** (no lexical-only degraded publish). Atomic from the user's viewpoint.
8. Startup `reconcile`: fail stale `staging` versions; report/remove Chroma IDs not expected by SQLite; **never** mutate SQLite based on Chroma contents; if the collection is missing/short, run `rebuild`.
9. Write lock serializes ingestions; queries read a single SQLite snapshot (one statement joins candidates to `eligible_chunks` and returns chunk text), so a query never sees two versions of one document.

**Tests first (FakeEmbedder + real SQLite + real PersistentClient on `tmp_path`):**
1. add → query finds content
2. identical re-upload → `unchanged`, **0** embed calls; same parsed text with different bytes → also 0
3. local edit → only changed chunks embedded
4. after publish, old text is unretrievable
5. delete → unretrievable via lexical **and** dense, **with Chroma cleanup fault-injected to fail** (I2)
6. re-add → new version searchable; old version's IDs never reappear
7. embedder failure mid-update → previous version fully queryable (I3)
8. Chroma failure during staging → previous version fully queryable (I3)
9. kill between stage and publish, restart, `reconcile` → consistent
10. delete `data/chroma/`, run `rebuild` → dense search restored, **0** embed calls (I10)
11. concurrent query during update never returns mixed versions
12. duplicate chunks keep multiplicity; reorder-only edit → 0 embed calls

**Gate:** all 12 pass; `ingest_log` rows contain counts/timings but no text.

---

### P7 — Retrieval, evidence gate, extractive answering  `[L]`

**7.1 Retrieval** (`retrieval/service.py`)
1. Normalize/tokenize the question with the same functions as documents. Empty/whitespace question → `422`.
2. **Dense:** embed query (LRU-cached) → `VectorStore.query(n = CANDIDATES_N × OVERFETCH)`.
3. **Lexical:** FTS5 `bm25` over `eligible_chunks`, top `CANDIDATES_N`.
4. **Eligibility:** join dense IDs to `eligible_chunks` in one SQL statement (also applies `doc_ids` / `language` filters) and fetch chunk text + metadata in that same statement. Anything not returned is dropped (I1, I2). If filtering leaves fewer than `CANDIDATES_N` while Chroma returned a full page, re-query once with a larger `n` (bounded).
5. **Fusion:** weighted RRF `score = w_d/(k+rank_d) + w_l/(k+rank_l)`; keep raw dense similarity, bm25 score, and token coverage beside the fused score for the gate and debugging. Treat 0.7/0.3 as a *baseline to evaluate*, not a truth.

**7.2 Evidence gate** (`retrieval/gate.py`) — runs **before** any answer text is chosen.
- Signals: `max_dense_similarity`, normalized lexical score, `token_coverage` (share of non-stopword query tokens present in the best chunk), dense/lexical agreement (top-k overlap), optional top-1/top-2 margin.
- Decision rule (auditable, thresholds from `config/thresholds.json`, **keyed by `model_id`** because similarity distributions differ per model):
  `passed = (max_dense ≥ min_dense AND coverage ≥ min_coverage) OR (lexical ≥ min_lexical AND coverage ≥ min_coverage_high)`
- Output is a structure, e.g. `{"passed": true, "reason": "dense+coverage", "max_dense": 0.83, "token_coverage": 0.61, "thresholds_version": 1, "calibrated": true}`; returned only when `debug=true`.
- Ship conservative placeholders flagged `"calibrated": false` until P9; `/ready` reports this flag and the README must not claim calibration before P9 completes.

**7.3 Extractive answerer** (`answering/extractive.py`)
- Split retrieved chunks into sentences (reuse the P3 splitter, keeping char offsets relative to the chunk).
- Score = IDF-weighted query-token overlap (+ small retrieval-score term). A sentence must meet `min_sentence_overlap` to be eligible — never pad with unrelated sentences from a relevant chunk.
- Select ≤ `MAX_ANSWER_SENTENCES`, order by source position, merge adjacent selections into one contiguous slice.
- Output **segments**: `{text, citation_id, chunk_char_start, chunk_char_end}` where `text == chunk.text[start:end]`. The `answer` string is a *rendering* of the segments (`text [n]`); the `[n]` markers are system metadata, not content.
- **Never:** paraphrase, translate, add connectives, alter numbers/names/units/negations, or use world knowledge.
- **Insufficient evidence:** `status="insufficient_information"`, `citations=[]`, `answer` = a fixed localized system message (English: `Not enough information in the provided documents.`; Persian equivalent in a small message table), plus `reason` ∈ `{empty_knowledge_base, no_relevant_content, below_threshold}`. The message is not evidence and is never cited. Never show "closest" chunks as normal citations.
- `evidence_score ∈ [0,1]` is a deterministic evidence score, **not a probability** (API field name avoids `confidence` on purpose).

**7.4 Citation model**
```json
{"id": 1, "doc_id": "handbook", "document": "handbook.md", "doc_version": 3,
 "section": "Guide > Install > Linux", "page": null, "lines": [42, 58],
 "chunk_id": "handbook:v3:7", "char_span": [120, 214],
 "excerpt": "exact source text", "score": 0.83}
```
Sentence→citation mapping is stable (segment `citation_id`).

**Optional (only if P9 shows a gain):** dense re-scoring of candidate *sentences* (extra embedding requests per query; must respect the rate limiter and be behind `SENTENCE_RERANK=true`).

**Tests first:** answerable → evidence from the expected version; unanswerable (near-topic vocabulary, off-topic, empty KB) → `insufficient_information`, no citations; **every segment is an exact substring of its chunk and of `version.source_text` at the recorded offsets**; a mutation test that corrupts the answerer's output fails the substring test; deleted/outdated content never appears (answer, citations, or non-debug output); Persian question cites Persian source with correct section/page/line; gate threshold boundary cases; hostile FTS input is safe.

**Gate:** all green; I1, I2, I6, I7 tests pass through the public retrieval/answer entry point.

---

### P8 — API, observability, operations  `[M]`

| Method | Path | Behavior |
|---|---|---|
| POST | `/documents` | multipart `file` + optional `doc_id` (default: slug of filename). `201` created · `200` unchanged · `409` exists with different content (use PUT) · `413` · `415` · `422` |
| PUT | `/documents/{doc_id}` | Replace an **active** document → `200` (`updated` or `unchanged`) · `404` unknown/deleted |
| DELETE | `/documents/{doc_id}` | `204` · `404` unknown (repeat delete of a deleted doc → `404`) |
| GET | `/documents` | Active documents, paginated (`limit`, `offset`) |
| GET | `/documents/{doc_id}` | Metadata + current version + section outline · `404` |
| POST | `/query` | `{question, top_k?, doc_ids?, language?, debug?}` → `answered` \| `insufficient_information` |
| GET | `/health` | Liveness — always `200` |
| GET | `/ready` | Readiness — `200`/`503`: SQLite usable, FTS5 present, **local Chroma opens + lock held + dimension OK**, embedding model valid (cached; **no embedding call per request**), `thresholds_calibrated`, `embedder` provider |

Query response:
```json
{"status": "answered",
 "answer": "Exact source excerpt [1].",
 "segments": [{"text": "Exact source excerpt.", "citation_id": 1}],
 "evidence_score": 0.82,
 "citations": [ /* §7.4 */ ],
 "debug": null}
```
Error shape (one stable form): `{"error": {"code": "DOCUMENT_NOT_FOUND", "message": "...", "request_id": "..."}}`. Unknown exceptions → `500` without stack traces or secrets. Add a request-ID middleware; log ids, counts, timings, error classes only.

CLI (`python -m qasystem.cli`): `reconcile`, `rebuild`, `check-storage` (Chroma + SQLite round-trip: upsert/query/delete/count), `calibrate`, `eval`.

**Tests first:** FastAPI `TestClient` over every endpoint × happy path, validation path, domain error, and response shape; OpenAPI schema snapshot; upload-size and content-type handling; no secret in any error body.

**Gate:** all green; the server starts with `--workers 1`; a second instance on the same `data/` refuses to start (L2); `scripts/smoke_test.sh` passes (add → query → edit → query → delete → query).

---

### P9 — Evaluation, model selection, gate calibration  `[M]`

**9.1 Dataset** (`tests/eval/dataset.jsonl`, committed, deterministic). Build from the fixture corpus (≥ 6 documents, 2 formats each where possible). Target ≥ 50 questions:

| Type | Target count |
|---|---|
| Answerable, English | 15 |
| Answerable, Persian (A1) | 10 |
| Exact-term / identifier / number | 8 |
| Multi-section | 5 |
| **Unanswerable** — near-topic vocabulary | 8 |
| **Unanswerable** — off-topic / empty-ish | 4 |

Each answerable case uses a **chunking-independent gold**: `{"doc": "handbook.md", "must_contain": "exact phrase from source"}`. A retrieval hit = a retrieved chunk whose text contains `must_contain` (chunk IDs change whenever chunking is tuned, so never use them as gold). Split dev/test (e.g. 60/40) **before** looking at results.

**9.2 Metrics** (report for dense-only, lexical-only, hybrid): Recall@1/3/5, MRR@5, unanswerable false-answer rate, answerable refusal rate, citation substring validity (must be 100%), stale-content leakage (must be 0), P50/P95 query latency, index size, and ingestion embed-request count.

**9.3 Model selection.** For each *reachable* candidate (C4): fresh temp `DATA_DIR`, ingest corpus, run the eval, record metrics. Embedding cache makes re-runs free. Decision rubric, in order: (1) no invariant violation; (2) best Recall@3/MRR@5 on the dev split at an acceptable false-answer rate; (3) tie-breakers: smaller dimension (index size, speed), lower latency, fewer requests. Keep `BGE-M3` as the *prior* default (multilingual, 1024-d, compact) but **select by measurement** and log the decision in `docs/DECISIONS.md`. Make no pricing claims you cannot source.

**9.4 Gate calibration** (`make calibrate`). Grid-search thresholds on the **dev** split maximizing answerable recall subject to unanswerable false-answer rate ≤ 5% (prefer 0%, since the task is evidence-first); report the operating point on the **held-out** split. Write `config/thresholds.json` with `model_id`, `version`, `calibrated: true`, and the dataset size/class balance. State plainly that a dataset this small gives coarse estimates.

**9.5 Optional experiments (only if time):** RRF weights, chunk size, stemming, sentence reranking. Adopt a change only if it improves the dev metric without hurting held-out results.

**Gate:** `docs/eval_report.md` committed with real numbers; README quotes them; invariants still green.

---

### P10 — Hardening and README  `[S]`

- Run the full final checklist (§12) and fix gaps; coverage ≥ 85% on `ingestion/`, `retrieval/`, `answering/`, `text/`, `parsing/`.
- Add regression tests for every bug found.
- Write README (§10). Verify a clean clone works: `uv sync && cp .env.example .env && make check && make run`.

**Gate:** Definition of Done (§11) fully checked.

---

## 8. Testing Matrix

| Area | Required tests |
|---|---|
| Config/security | missing key, env loading, `SecretStr` redaction, `fake` provider blocked in prod |
| Text | Persian variants, digits, ZWNJ, idempotence, identifiers, mixed scripts |
| Parsers | MD headings/fences/Setext, TXT encodings, PDF pages, corrupt/scanned/encrypted PDF, size limits |
| Chunking | determinism, bounds, source coverage, slice identity, local-edit stability |
| Embeddings | order, batching, limiter, `Retry-After`, 429/5xx, auth, dimension, model-not-listed, cache |
| Storage | SQLite rollback/idempotent schema, `eligible_chunks`, hostile FTS input, **Chroma restart persistence**, cosine conversion, lock, dimension isolation |
| Diff | reused/added/removed, reorder, duplicates (multiset) |
| Ingestion | add/unchanged/edit/delete/re-add, failure injection (embedder, Chroma), crash-before-publish, rebuild with 0 API calls, concurrency |
| Retrieval | dense-only, lexical-only, fusion, eligibility, doc/language filter, over-fetch |
| Gate/answer | answerable/unanswerable, threshold boundaries, verbatim-only, citation mapping, mutation test |
| API | every endpoint, status codes, validation, error shape, OpenAPI snapshot |
| Eval | runner, metric computation, reproducible report |

Invariant tests (I1–I10) are release blockers. Coverage is a signal, not a substitute for them.

---

## 9. Coding Standards for AI Agents

1. Type hints everywhere; `mypy src` clean. Prefer small pure functions and explicit dependency injection.
2. Network and storage I/O live behind ports/adapters; no global mutable state for clients/repositories/config.
3. No bare `except:`. Convert low-level exceptions to typed domain errors at architectural boundaries.
4. `async` for external network I/O; wrap blocking SQLite/Chroma work in `asyncio.to_thread`.
5. Log identifiers, counts, timings, error classes — never document bodies or secrets.
6. Every tunable belongs in config. No magic thresholds inline.
7. Every bug fix adds a regression test; every behavior change updates README/OpenAPI/tests in the same commit.
8. Never silently fall back to another model, another version, or a fabricated answer.
9. Don't add abstractions until a second implementation or a test seam requires it.

---

## 10. README Requirements

1. Problem statement, constraints, assumption A1.
2. Architecture diagram (§3) and why there is **no LLM generation** (embedding-only service).
3. Model-selection experiment and final choice (link `docs/eval_report.md`).
4. Storage design: SQLite source of truth · FTS5 lexical · **local persistent Chroma** as a disposable derived index; data directory layout; backup (stop → copy `data/`) and `rebuild`.
5. Version publication and deletion consistency model (publish-then-cleanup, `eligible_chunks`).
6. Chunking and Persian-normalization decisions.
7. Retrieval, evidence-gate, and answer-selection methodology; meaning of `evidence_score`.
8. Evaluation dataset, measured metrics, calibration procedure and its limits.
9. API usage examples (`curl`) for upload / update / delete / query.
10. Setup (`uv`, env vars, **single-worker requirement**) and test instructions.
11. Known limitations: no OCR/scanned PDFs; extractive answers can't synthesize across distant passages; single-process deployment; small eval set; PyMuPDF licence note.
12. Security notes: token externalized, redacted, never committed.

---

## 11. Definition of Done

- [ ] PDF, TXT, Markdown ingestion works (English fixtures; Persian per A1).
- [ ] Add / update / delete via API; identical re-upload → zero embedding requests.
- [ ] Local edits re-embed only changed chunks; reorder-only edits re-embed nothing.
- [ ] Deleted and superseded content provably unretrievable (I1–I3), including with cleanup fault-injected.
- [ ] Failed updates never expose a partial version (I3).
- [ ] Chroma runs **local/persistent**; restart persistence verified; single-instance lock enforced; `rebuild` restores dense search with zero API calls (I10).
- [ ] Hybrid retrieval measured against dense-only and lexical-only baselines.
- [ ] Answers are exact source excerpts; every citation mechanically verified (I6, I7).
- [ ] Unanswerable questions → `insufficient_information`, no citations.
- [ ] No GUI dependency; stable OpenAPI; structured errors; no secrets exposed (I8).
- [ ] Unit, integration, failure-injection, concurrency tests green; `make check` clean.
- [ ] `docs/eval_report.md`, thresholds file, and `docs/DECISIONS.md` committed; README complete and honest about limits.

---

## 12. Final Engineering Review Checklist (answer from tests, not assumptions)

- If Chroma still holds a deleted document's vectors, does the app provably filter them out?
- Does a failed update leave the old version fully queryable?
- Does an unchanged upload make zero embedding calls? A reorder-only edit?
- Does a paragraph edit preserve most unaffected chunk hashes? Are duplicate chunks handled with multiplicity?
- After deleting `data/chroma/`, does `rebuild` restore search with no API calls?
- Does a second process on the same `data/` fail fast instead of corrupting Chroma?
- Can a Persian query retrieve and cite a Persian source without manual RTL handling?
- Is user text ever interpreted as FTS5 syntax?
- Can any answer segment be shown to be an exact substring of stored source text?
- Do unanswerable questions avoid returning misleading "closest" evidence as normal citations?
- Is `evidence_score` documented as a non-probability?
- Can the embedding model change without mixing dimensions/collections, and are thresholds model-specific?
- Is the token absent from source control, logs, and error bodies?