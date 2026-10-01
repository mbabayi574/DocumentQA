# Document QA System — LLM-free Extractive RAG (v4)

> **For:** AI coding agents, and the human reviewing `jobTask.md`.
> **Read:** §0–§5 plus only the phase you are running. Stop when its Gate passes.
>
> **v4 changes.** BGE-M3 is the only model in this document; `README.md` is the sole place
> any other model is named, to justify that selection. Per-version deviation tables are
> gone — `docs/DECISIONS.md` (D1–D31) holds that evidence. Status, measurements, and
> hard-won constraints from P0–P5 are folded in, so this file no longer describes a system
> that does not exist.

**Two facts govern everything else.**

1. The provider is **embedding-only** — `GET /v1/models`, `POST /v1/embeddings`, no chat
   endpoint. So the system is **extractive**: retrieval finds evidence, the answerer
   returns **verbatim source slices with citations**. Nothing paraphrases, summarizes,
   translates, or completes an answer.
2. The model is **BGE-M3** (id `Bge-m3`, 1024 dimensions). §2 lists its measured
   properties and the implementation decision each one forces. Optimise *for BGE-M3*.

---

## 0. Operating protocol

1. **Test first.** Write the phase's *Tests first* list, watch it fail, implement, refactor.
2. **Gate before commit.** `make check` (ruff check + format, `mypy src` strict, `pytest -q`)
   green, **`make live` green against the real provider**, the phase's Gate items pass, **and** a
   fresh `git clone` passes too (§0.11).
   > `make live` is not optional and not a nice-to-have. The fake embedder is a bag of hashed
   > tokens: it cannot rank, cannot align languages, and its similarity scale is nothing like
   > the real model's. P0-P7 passed 480 offline tests while the gate was **unable to authorise
   > the system's own cross-lingual capability** (D47) and a Latin word glued to Persian script
   > was **unmatchable** (D45). Four defects, none of them visible offline. Offline green proves
   > the plumbing; only live proves the product.
3. **Never weaken a test to get green.** If a test looks wrong, log it in
   `docs/DECISIONS.md`, flag it, and move on.
4. **Measure, don't recall.** Any third-party behaviour the code depends on gets a probe
   test. If you find an assumption is wrong, correct the plan and log the evidence.
5. **Mutation-check the important ones.** For each invariant and each safety property,
   break the code on purpose and confirm a named test fails. A test that passes for the
   wrong reason is worse than no test (D25, D26).
6. **No network in tests.** `socket.connect` is blocked in `tests/conftest.py`. Live calls only
   under `RUN_LIVE=1` (`scripts/measure_provider.py`, `tests/integration/*_live.py`). Every phase
   adds its live coverage there rather than assuming the offline suite carries it.
7. **Secrets.** `EMBEDDING_API_KEY` from env only. Never logged, printed, returned, or
   committed — including inside error bodies built from provider responses (I8, D23).
8. **Scope guard — do NOT add:** OCR, auth, GUI, message queues, any LLM/chat call, another
   vector database, Chroma HTTP/server mode, or a silent model/version fallback.
9. **Stop after 3 failed attempts** on one test: report hypothesis and evidence, ask.
10. **Commit small** — one logical change, tests included. Record decisions with evidence
    in `docs/DECISIONS.md`.
11. **Verify a clean clone.** `make check` in your working tree proves nothing about the
    *repository*; a fixture that exists locally but is untracked leaves the tree green and
    every clone red (D31). `tests/unit/test_repo_consistency.py` now guards this.
12. **Keep chat output small.** Never dump embeddings, whole documents, or whole files.

**Order.** One phase per session, one commit per logical change, in dependency order.
P6 was the highest risk, and is done; P7 is done. P8 (API) is next.

```text
P0 ✓ scaffold    P1 ✓ text     P2 ✓ parsing   P3 ✓ chunking   P4 ✓ embeddings   P5 ✓ storage
→ P6 ✓ ingestion → P7 ✓ retrieval+gate+answer → P8 ✓ API (incl. whole-project live)
→ P9 eval + calibration → P10 hardening → P11 refactor + README
```

---

## 1. `jobTask.md` requirement → design → proof

| # | Requirement | Design | Proof |
|---|---|---|---|
| 1 | Process PDF, TEXT, Markdown for retrieval | One parser per format → `ParsedDocument` (source text + char/page/line offsets). Normalised tokens feed FTS5; chunk slices feed the embedder. **Stored text is never rewritten**, so citations stay exact. | P1–P2 unit tests per format; P3 slice-identity tests |
| 2 | Add/edit/delete; never use outdated content | Monotonic versions per `doc_id`. Publication is **one SQLite transaction** flipping `current_version`. One `eligible_chunks` view gates **every** retrieval path, so stale and deleted rows are unreachable even if Chroma cleanup never ran. Chroma is derived and disposable. | P6 tests 1–12: idempotency, edit isolation, delete with cleanup fault-injected, failure injection, crash recovery |
| 3 | Relevant retrieval | Hybrid: dense (local Chroma, cosine) + lexical (FTS5 `bm25`), fused by weighted RRF over the **candidate union** (§9.1). Candidate scores carry raw dense similarity, `bm25`, and token coverage for the gate. | P7 fusion/eligibility tests; P9 R@1/3/5 and MRR@5 for dense-only vs lexical-only vs hybrid |
| 4 | Evidence-based, traceable answers | Extractive answerer returns source slices only. Citations carry doc, version, section path, page, lines, chunk id, char span, exact excerpt. | P7: every segment is an exact substring of its chunk **and** of `source_text` at the recorded offsets; mutation test proves the check bites |
| 5 | Say so when information is insufficient | An evidence gate runs **before** any answer text is selected. Below threshold → `status="insufficient_information"`, `citations=[]`, a fixed localised system message, a machine-readable `reason`. "Closest" chunks are never presented as citations. | P7 unanswerable tests (near-topic, off-topic, empty KB); P9 false-answer rate ≤5%, preference 0% |
| 6 | API access, no GUI | FastAPI + OpenAPI: upload/update/delete/query/health/ready, plus a maintenance CLI. No GUI dependency. | P8 `TestClient` per endpoint; `scripts/smoke_test.sh` |
| 7 | Readability, separation of concerns, error handling, tests | `api → services → domain/ports ← adapters`. Typed errors with stable codes and HTTP mapping. A port exists only where a second implementation or a test seam needs one. I1–I10 are release-blocking tests. | `make check`; clean clone green; coverage ≥85% on `text/ parsing/ chunking/ ingestion/ retrieval/ answering/`; P11 readability pass |

**Model selection** is settled and is documented in `README.md`. This plan carries only
BGE-M3's properties and what they force (§2).

---

## 2. BGE-M3 — measured properties and what each one forces

Every value was measured on this machine against the live service. Re-run
`RUN_LIVE=1 uv run python scripts/measure_provider.py` before trusting any of it.

| Measured property of BGE-M3 | Forced implementation decision |
|---|---|
| **1024 dimensions**, probed at startup | Never hard-code the dimension; discover it with one call. Assert against `EMBEDDING_DIMENSION` if set. Store it in the collection name and metadata so dimensions can never mix (I9). |
| **Vectors arrive L2-normalized** (‖v‖ = 1.000000) | Ingest skips normalizing. A probe test asserts ‖v‖≈1 so a provider change is caught rather than silently skewing cosine. |
| **40 949-char single-item ceiling** (binary search, 15 probes) | `MAX_CHARS_PER_ITEM=20000` → **2.05× margin**. Enforced *before* any request, so an oversized chunk fails locally rather than burning quota on a predictable 400 (D22). |
| **32 items/request is the throughput peak** (29.7 chunks/s; 23.6 at 64, 28.2 at 128) | `MAX_ITEMS_PER_BATCH=32`. Do not "optimise" this upward. |
| **Cost per character falls steeply with item length**: 105 ms per 1 000 chars at 500-char items, 36 ms at 2 000, 11 ms at 8 000 — **2.9x** then **9.5x** cheaper than 500 | Chunk size is the biggest cost lever, but it trades against retrieval precision. **P9 experiment, not a change** — `CHARS_PER_TOKEN=1.5` sizes for the densest text (tables), so real chunks average ~500 chars against a 40 949-char budget. |
| **120 req/min; real ingest needs 31 req/min** | `RATE_LIMIT_PER_MIN=100` leaves headroom. Ingest cost is dominated by request count, not characters. |
| **Multilingual: English question → Persian source with no translation step** — **verified live, with a cost** | No language routing, no translation, no per-language collection. One collection serves both languages; `detect_language` is a stored label, not a routing decision. Live-verified: an English question retrieves and is answered from the Persian PDF. **But** `token_coverage` counts *shared tokens*, so coverage is structurally 0 for a cross-lingual hit — the gate needs an uncorroborated `dense_only` branch (D47) and the answerer needs a positional fallback (D48), and that branch carries a measured **1-in-9 false-answer rate** the cosine cannot separate from a real one (D47). Measured again on the full corpus: cross-lingual retrieval then **finds the right chunk and loses it to the lexical arm** — the correct Persian chunk is dense rank 2 at similarity 0.539 with *zero* lexical contribution, while English chunks that share incidental words collect lexical ranks 1–19 at similarity 0.547 and overtake it. No threshold separates them (D55). See §12. |
| **The `index` response field is a permutation, but the API does not guarantee it** | Validate it is a permutation of `0..n-1`; otherwise use response order and log a warning. Cheap insurance against silently mislabelled vectors (D21). |
| **`Retry-After` is absent on 429** | Capped exponential backoff with jitter is the real path. Honour the header if a provider ever sends it. |
| **403 for a model the account cannot use**, with the allowed IDs in the message | `EmbeddingAuthError` carries the allowed list. 401/403 and other 4xx fail fast — never retried. |
| **A whole fixture corpus ingests in 33 requests** | `rebuild` reads `embedding_cache` and re-upserts, so restoring dense search costs **zero** API calls (I10). |

**Retrieval reality check (D30).** On the committed corpus, dense-only is
R@1 0.67 / R@3 0.83 / R@5 0.92, lexical-only R@1 0.42, hybrid R@3 0.92. Two consequences:
dense similarity alone is **not** accurate enough to answer from, which is what makes the
gate load-bearing; and **`DENSE_WEIGHT=0.7` is a starting point, not a tuned constant** —
at 12 questions, one question is worth 0.083 R@1, so P9 must decide it on the real
dataset. An earlier `R@1 = 1.000` figure did not reproduce and is withdrawn.

---

## 3. Environment (measured; code may assume nothing else)

**SQLite 3.53.1, FTS5 present** — checked at startup, not assumed. Pragmas
`journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout`.

**ChromaDB 1.5.9, local only.** `PersistentClient(path=…, Settings(anonymized_telemetry=False))`.
Cosine via `metadata={"hnsw:space":"cosine"}`. Measured: identical vector → distance `0.0`,
orthogonal → `1.0`, so **similarity = 1 − distance**. `n_results` above `count()` is
silently clamped, so clamp explicitly and skip the query when the count is 0. An empty
collection returns empty lists rather than raising. `heartbeat()` returns an int timestamp.
Collection metadata survives a client restart.

> **Chroma caches per path for the life of the process** (D28, D29). Deleting
> `data/chroma/` is only observable from a *new* process, so test `rebuild` as
> "restore into an empty index" and cover persistence with a separate reopen test.

**`filelock` 4.0.7** — `FileLock(path, timeout=0)` raises `Timeout` in a second process, so
L2 is enforceable. Verified with real subprocesses; a same-process check is weaker than it
looks because a temporary holding the lock is garbage-collected and releases it.

**Provider request shape.** `GET /v1/models` → `{"data":[{"id":…}]}`; the configured
`EMBEDDING_MODEL` must match an id **exactly** (`Bge-m3`, not `BGE-M3`). 200 000 chars per
request; empty `input: []` is a 400. `.env` uses CRLF; `pydantic-settings` strips it.

**Python ≥3.12**, pinned deps, `uv`. `numpy` arrives transitively via `chromadb` and is
imported nowhere; it stays in the domain-purity test's forbidden list as a string.

**Fixtures** (`tests/fixtures/docs/`, all committed except one):

| File | Role |
|---|---|
| `en/storyen.md`, `fa/storyfa.md` | Markdown, en + fa (ZWNJ) |
| `en/clean-code-excerpt.pdf` | Clean English text layer; built by `scripts/build_pdf_fixtures.py` |
| `fa/ai-engineer.pdf` | Persian PDF, page 2 blank — proves one blank page does not void a document |
| `fa/justforfun_book_a4.pdf` (749 KB) | **NFKC is required**; raw glyphs are presentation forms. Also pins the lost-space limitation below. 84% of the index — see §2's retrieval reality check. |
| `en/Clean Code Fundamentals…pdf` (10 MB) | Deliberately **not** committed; its test skips when absent |

**Known limitation, stated not hidden.** The Persian book's font maps the space glyph to
nothing on 274 of 6 563 content lines, so those arrive as one run of letters. NFKC recovers
letters and order; it cannot invent a space the PDF never stored. Repairing it needs
Persian word segmentation — out of scope. Retrieval still works on the characters present.

**Assumption A1 — Persian.** `jobTask.md` does not mention languages; bilingual is assumed
from the provider's locale, English first, Persian as a bounded second stream, and the P9
eval corpus is bilingual by decision. Some Persian PDFs in the wild have unrepairable text
layers; we detect and report them rather than emit garbage. **No OCR** (§0.8).

---

## 4. Architecture

```text
            +-------------------------------+
            |            FastAPI            |   single process, 1 worker
            | upload / update / delete /    |
            | query / health / ready        |
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

**Load-bearing decisions.**

- **SQLite is the single source of truth.** Publication state exists nowhere else.
- **Chroma is derived and rebuildable.** Vectors and three diagnostic fields only, never
  document text. `embedding_cache` retains every vector, so `rebuild` needs no API.
- **Publish is one transaction.** No second "is active" flag; eligibility is derived
  through a view, never remembered.
- **Versions are monotonic per `doc_id`, across delete/re-add.** Vector ids are
  `"{doc_id}:v{version}:{ordinal}"`, so a stale vector can never collide with a fresh one.
- **Publish, then clean up.** An incomplete update is ignorable; a published one is
  complete. Cleanup is best-effort and its failure must not change what is retrievable.
- No message queue: the API shape does not need one.

### 4.1 Local Chroma rules

| # | Rule |
|---|---|
| L1 | `PersistentClient` only. **Never** `HttpClient`, `EphemeralClient`, or an in-memory fallback outside tests. |
| L2 | **Single owner.** `uvicorn --workers 1`; at startup take `FileLock(data/.qasystem.lock, timeout=0)`. If held, fail fast with a clear message. |
| L3 | Chroma is blocking: wrap in `asyncio.to_thread`. Serialise Chroma + SQLite writes behind one application write lock. |
| L4 | Collection name `chunks__{slug(model_id)}__d{dimension}` (`[a-z0-9_-]`, <63 chars). Store `model_id`/`dimension` in metadata and **assert on open** (I9) — the name can collide after truncation, so metadata is the real guard (D27). |
| L5 | Cosine space on the pinned version; `similarity = 1 − distance`, proven by a known-vector test. |
| L6 | **No document text in Chroma.** Metadata: `doc_id`, `doc_version`, `ordinal`. SQLite decides eligibility. |
| L7 | Clamp `n` to `count()`; skip the query when the count is 0; over-fetch so filtered-out stale vectors cannot starve the list. |
| L8 | `data/` is git-ignored. **Backup = stop the app, copy `data/`.** Never mutate Chroma's files directly. |
| L9 | Readiness: data dir writable, lock held, `heartbeat()`, collection opens, dimension matches. Report failures; never silently recreate a different store. |
| L10 | Tests use a real `PersistentClient` on `tmp_path`. Restart test: write → drop the client → reopen → still found. |

---

## 5. Invariants, errors, and the data model

### 5.1 Release blockers — each one a test

| ID | Invariant | Enforced by |
|---|---|---|
| I1 | Active version: every evidence item comes from a chunk whose `doc_version == documents.current_version` of an `active` document | `eligible_chunks` view |
| I2 | Deletion: after the delete transaction commits, the document is unretrievable lexically **and** densely, **even if Chroma cleanup never ran** | the same view; test with cleanup fault-injected |
| I3 | Atomic publication: a failure before publish leaves the previous version fully queryable and unchanged | one explicit transaction per operation |
| I4 | Idempotency: identical bytes — or identical parsed text — returns `unchanged` with **zero** embedding requests and no new version | `parsed_hash` early-exit |
| I5 | Reuse: unchanged chunks cause zero embedding requests | cache keyed by `sha256(exact embedded input)` |
| I6 | Traceability: `source_text[char_start:char_end] == chunk.text`, and every answer segment is an exact substring of its cited chunk | contiguous-slice chunking + substring tests |
| I7 | No synthesis: answer text is selected source slices in source order, `[n]` markers rendered outside the slices | answerer + substring tests |
| I8 | Secrets: the token never appears in logs, errors, snapshots, fixtures, or committed files | `SecretStr` + `scrub()` on provider messages |
| I9 | Model isolation: one collection per `(model_id, dimension)`; vectors of different models or dimensions never mix | L4 name + metadata assertion |
| I10 | Disposable index: deleting `data/chroma/` and running `rebuild` restores dense search with **zero** API calls | `rebuild` reads `embedding_cache` only |

> **D25, D26 — the two ways these lie.** `with self._db:` is a **no-op** under
> `isolation_level=None`: it commits each statement separately, so nothing rolls back. Use
> an explicit `BEGIN`/`COMMIT`/`ROLLBACK`. And the view's `status='active'` guard is
> invisible while `mark_deleted` also nulls `current_version` — so deleting that guard
> passes every test. Test each guard against the state it alone defends.

### 5.2 Ports (`src/qasystem/domain/`)

Pure stdlib: no FastAPI, Chroma, httpx, or sqlite3 — enforced by an AST test. Shapes live
in `domain/models.py` and `domain/ports.py`; they are frozen and implemented.

- `Embedder` — `model_id`, `dimension`, `async embed(texts) -> list[list[float]]`
  (order-preserving).
- `VectorStore` — `ensure_collection()`, `upsert`, `query`, `get_existing_ids`, `list_ids`
  (paged), `delete_ids`, `count`, `ping`. **`ensure_collection` takes no arguments on
  purpose** (D27): a store is constructed for one `(model_id, dimension)`, and
  re-supplying them invites opening the wrong collection.
- `DocumentParser` — `parse(data, filename) -> ParsedDocument`.

Application *services* (`IngestionService`, `RetrievalService`, `AnswerService`) are not
ports. **A new Protocol is added only when a second implementation or a test seam
requires it.**

### 5.3 Typed errors

Each has a stable `code` and HTTP mapping. `InsufficientEvidence` is **not** an error — it
is a normal `200`.

| Error | Code | HTTP |
|---|---|---|
| `UnsupportedFormatError` | `UNSUPPORTED_FORMAT` | 415 |
| `FileTooLargeError` | `FILE_TOO_LARGE` | 413 |
| `EmptyDocumentError` / `ParseError` / `NoTextLayerError` | `EMPTY_DOCUMENT` / `PARSE_ERROR` / `NO_TEXT_LAYER` | 422 |
| `DocumentNotFoundError` | `DOCUMENT_NOT_FOUND` | 404 |
| `DocumentExistsError` | `DOCUMENT_EXISTS` | 409 |
| `EmbeddingUnavailableError` (retries exhausted) | `EMBEDDING_UNAVAILABLE` | 503 |
| `EmbeddingAuthError` (401/403, incl. model-not-available) | `EMBEDDING_AUTH` | 502 |
| `VectorStoreError` / `StorageLockedError` | `VECTOR_STORE_UNAVAILABLE` / `STORAGE_LOCKED` | 503 |
| `ConfigError` | `CONFIG_ERROR` | 500 |
| anything else | `INTERNAL_ERROR` | 500, no stack trace, no secrets |

### 5.4 Data model

`src/qasystem/storage/schema.sql` is the source of truth: idempotent, `PRAGMA user_version`
for migrations, with `documents`, `document_versions`, `chunks`, `chunks_fts`,
`embedding_cache`, `ingest_log`, and the view below. Read it, do not re-describe it.

```sql
-- THE choke point. Every retrieval path reads chunks ONLY through this view (I1, I2).
CREATE VIEW eligible_chunks AS
  SELECT c.* FROM chunks c
  JOIN documents d ON d.doc_id = c.doc_id
                  AND d.status = 'active'
                  AND d.current_version = c.doc_version;
```

`chunk_id` doubles as the Chroma vector id. `embed_input_hash` is the sha256 of the exact
embedded string, which is what makes I5 and I10 work. `ingest_log` carries counts and
timings only — never bodies, text, or secrets. Lexical search indexes **our** tokenizer's
output, ZWNJ components included, so SQLite never has to understand Persian.

---

## 6. Configuration

Every tunable lives in `config.py` and is validated at startup; none is inline. `.env` is
git-ignored; `.env.example` holds placeholders. Current values and the reason for each are
in the file's comments; the ones with measured justification:

```dotenv
EMBEDDING_MODEL=Bge-m3               # exact id from GET /v1/models
EMBEDDING_DIMENSION=                 # optional; asserted against the probed 1024
EMBEDDING_PROVIDER=remote            # remote | fake (fake only when APP_ENV is test/demo)
CHUNK_TARGET_TOKENS=350              # × CHARS_PER_TOKEN=1.5 -> 525 chars
CHUNK_HARD_MAX_TOKENS=700            # -> 1050 chars, the hard cap P3 enforces
MAX_CHARS_PER_REQUEST=160000         # provider cap is 200000
MAX_CHARS_PER_ITEM=20000             # measured ceiling 40949 -> 2.05x margin
MAX_ITEMS_PER_BATCH=32               # measured throughput peak
RATE_LIMIT_PER_MIN=100               # measured ceiling 120; real ingest needs 31
CANDIDATES_N=30  OVERFETCH=2  TOP_K=5  RRF_K=60
DENSE_WEIGHT=0.7  LEXICAL_WEIGHT=0.3 # starting point only; P9 decides (see §2)
# config/thresholds.json (keyed on model_id; a mismatched file is refused):
GATE min_dense=0.47  min_dense_alone=0.50  min_coverage=0.25  min_lexical=0.80
#   min_coverage_high=0.50  min_sentence_overlap=0.15   calibrated=false
#   Measured live: answerable max_dense 0.501-0.631, unanswerable 0.359-0.451.
#   The previous guess of 0.62 sat INSIDE the answerable range and was dead (D47).
```

---

## 7. Repository layout

```text
├── README.md  plan.md  AGENTS.md  Makefile  pyproject.toml  .env.example  .gitignore
├── config/thresholds.json            # written by `make calibrate`, committed
├── docs/ DECISIONS.md  eval_report.md
├── scripts/ smoke_test.sh  eval.sh  measure_provider.py  build_pdf_fixtures.py
├── src/qasystem/
│   ├── config.py  errors.py  logging_setup.py  cli.py
│   ├── domain/     models.py  ports.py
│   ├── text/       normalize.py  tokenize.py  language.py
│   ├── parsing/    base.py  markdown.py  text.py  pdf.py  registry.py
│   ├── chunking/   chunker.py
│   ├── embeddings/ client.py  rate_limit.py  caching.py  fake.py
│   ├── storage/    schema.sql  sqlite_store.py  lexical.py  chroma_store.py  lock.py
│   ├── ingestion/  diff.py  service.py  reconcile.py
│   ├── retrieval/  fusion.py  gate.py  service.py
│   ├── answering/  sentences.py  extractive.py
│   └── api/        app.py  routes.py  schemas.py  deps.py
├── tests/ unit/ integration/ api/ eval/{dataset.jsonl,runner.py,corpus/} fixtures/docs/{en,fa}/
└── data/                               # git-ignored: qasystem.db, chroma/, .qasystem.lock
```

Dependency direction: `api → services → domain/ports ← adapters`.

---

## 8. Completed phases

Gates passed; `docs/DECISIONS.md` has the evidence. These constraints must not regress.

| Phase | Delivered | Hard-won constraint to preserve |
|---|---|---|
| **P0** scaffold | pinned deps, config with `SecretStr`, typed errors, `scrub()` redaction, frozen contracts, socket guard | token never in repr, logs, or error bodies |
| **P1** text | NFKC + Persian letter/digit folding; ZWNJ **kept**, and its compound/fused/parts all emitted so a ZWNJ document matches a spaced query (D8); one stopword set; `detect_language` | hidden characters are Unicode category `Cf` — one category test beats a hand-listed table, but **ZWNJ and ZWJ must stay in the keep-set** (a D20 regression deleted ZWNJ until a test caught it) |
| **P2** parsing | md/txt/pdf → `ParsedDocument`, offsets index the parser's own text; NFKC in the PDF parser (D14); registry enforces size before parsing | PDF applies NFKC because we *construct* that text; no RTL reversal — mangling a citation is worse than saying so. Every PDF in `tests/fixtures/docs/` is parametrized, so a new fixture is covered automatically |
| **P3** chunking | contiguous slices only, deterministic, heading-aware, sentence packing + a mechanical cap pass | `CHARS_PER_TOKEN=1.5` is the **densest** measured case (tables 1.45, prose 4.14), not the average; the second cap pass exists for content with no sentence terminators at all |
| **P4** embeddings | discovery, dimension probe, three-bound batching, token bucket, bounded retry, caching | `index` trusted only if a permutation (D21); the per-item cap is enforced locally, before any request (D22); scrub provider messages before they reach an exception (D23) |
| **P5** storage | schema, repository, FTS5 lexical, Chroma adapter, reconcile + `rebuild`, single-owner lock | explicit `BEGIN`/`COMMIT` (D25); `ensure_collection()` takes no args (D27); `rebuild` accepts a tripwire embedder it never calls, so "zero API calls" is asserted rather than claimed (D28) |
| **P7** retrieval + gate + answering | `fusion.py` weighted RRF over the **candidate union**, `gate.py` auditable two-branch rule keyed on `model_id`, `service.py` retrieve → gate → answer, `sentences.py` + `extractive.py` selection only | `normalize_for_index` was **deleting newlines** (category `Cc`), fusing the last word of every line onto the first word of the next — every FTS token and every `chunk_hash` for multi-line text was wrong (D37); ranks are 1-based everywhere, so RRF never divides by `k` for the best hit (D38); a weight of **0 disables** an arm, or §9.2's single-arm baselines would silently mix two systems (D39); the gate judges the whole window and only `top_k` sizes the answer (D40); coverage is the max over candidates, so a fusion bug cannot present as a gate refusal (D41); thresholds for another model are a hard `ConfigError`, a missing file only falls back to uncalibrated defaults (D42); Persian coverage loses 0.29 to Ezafe suffixes, the largest known weakness of the lexical arm (D43) |
| **P8** API + operations | `api/deps.py` builds the whole graph once and owns L2/L9; `api/routes.py` thin verbs; `api/app.py` lifespan + one error envelope + request-id middleware; `api/schemas.py` is the OpenAPI contract; `cli.py`; `scripts/smoke_test.sh` | SQLite connections were **thread-bound**, so a store built in the lifespan and used from a request thread raised `ProgrammingError` and **every endpoint 500'd** — invisible to 480 offline tests (D49); `filelock.is_locked` is thread-local, so `/ready` reported the lock unheld and 503'd forever (D50); `SQLITE_PATH`/`CHROMA_PATH` were configured and silently ignored (D52); the `Embedder` port declared identity as mutable when every implementation exposes it read-only |
| **P8-live** whole-project live verification | `tests/integration/test_system_live.py`: the full 1225-chunk fixture corpus, the CLI against the real provider, I10 measured on real vectors | **no new defects** — the gaps were verification gaps, not code gaps. But two measurements P9 needs and no offline test can produce: query latency is **one embedding call** (13–21ms of our work against 286–1230ms of network, P50 410ms / P95 1.8s), and the cross-lingual miss is caused by the **lexical arm**, not the gate — the correct Persian chunk is dense rank 2 at sim 0.539 with zero lexical contribution while wrong English chunks at 0.547 collect lexical ranks 1–19 and overtake it (D54, D55) |
| **P7-live** live verification | 14 live tests (`RUN_LIVE=1`), FakeEmbedder → real BGE-M3 | a Latin word glued to Persian script was **unmatchable** — `\w` spans both scripts, so `embedding` never existed as a token, breaking the lexical arm and the coverage signal (D45); `embed_requests` counted `embed()` calls not HTTP requests, a **32× error** in the number §2 and §9.2 report (D46); the gate **could not authorise cross-lingual retrieval** because coverage is structurally 0 there, so the dense arm needed an uncorroborated branch (D47) — and then the *answerer* refused the same evidence for the same reason until it got a positional fallback (D48); the shipped `min_dense = 0.62` was measured to sit **inside** the answerable range, so the dense branch was dead (D47) |
| **P6** ingestion | `diff.py` multiset diff, `service.py` add/replace/delete/reconcile, all 12 gate tests | embedding runs **outside** the write lock and before any version exists, so an embed failure costs nothing (D33); VERIFY refuses a partial vector write; a failed version **keeps** its row as `failed` while a superseded one is fully purged (D35); a best-effort cleanup step needs its own assertions, because no correctness test can fail when it is deleted (D34); `embed_requests` is a delta and `reconcile` returns the plan it **found** (D36) |

---

## 9. Remaining phases

Run one per session. Do not start a phase before the previous Gate passed.

### P6 — Ingestion and change management `[highest risk]` — **DONE, see §8**

```text
validate → parse → chunk → content/parsed hash → unchanged? ──yes──► 200 "unchanged" (0 embed calls)
                                   │no
                      diff vs published chunks (hash → queue of prior chunks)
                                   ▼
            resolve cached embeddings → embed ONLY missing inputs
                                   ▼
   STAGE (SQLite txn): version state='staging' + chunk rows + FTS rows
   STAGE (Chroma):    upsert under "{doc_id}:v{N}:{ordinal}"
   VERIFY:            get_existing_ids(expected) == expected
                                   ▼
   PUBLISH (ONE SQLite txn): version→published, previous→superseded,
           documents.current_version=N, status='active', hashes updated
                                   ▼
   CLEANUP (best-effort, separate step): superseded chunk/FTS rows + Chroma ids
```

**Rules**

1. Never touch the published version before every embedding is resolved and staging is
   verified.
2. Allocate the version from `documents.last_version + 1` inside a transaction; never reuse.
3. Diff by `chunk_hash` with **multiset** matching (hash → queue) so duplicates keep
   multiplicity. A reorder alone must cost zero embed calls. "Reused" means the
   *embedding* is reused from cache and re-upserted under the new versioned id — local
   Chroma upserts are free.
4. Failure before publish → version `failed`, staged rows and vectors removed best-effort,
   previous version untouched (I3). Stragglers are swept by `reconcile`.
5. **Delete:** one transaction sets `status='deleted'`, `current_version=NULL`. From that
   commit the document is unretrievable (I2). Purge best-effort; keep the `documents`
   tombstone so `last_version` stays monotonic.
6. Re-add after delete creates version `last_version+1`.
7. If Chroma staging fails, **abort**. No lexical-only degraded publish.
8. Startup `reconcile`: fail stale `staging` versions; remove Chroma ids SQLite does not
   expect; **never** mutate SQLite based on Chroma's contents; if the collection is short,
   run `rebuild`.
9. One write lock serialises ingestions. A query reads a single SQLite snapshot —
   candidates joined to `eligible_chunks`, returning chunk text in the same statement — so
   a query never sees two versions of one document.

**Tests first** — `FakeEmbedder` + real SQLite + real `PersistentClient` on `tmp_path`:

1. add → query finds the content
2. identical re-upload → `unchanged`, **0** embed calls; identical parsed text from
   different bytes → also 0
3. local edit → only changed chunks embedded
4. after publish, old text is unretrievable
5. delete → unretrievable lexically **and** densely, **with Chroma cleanup fault-injected to
   fail** (I2)
6. re-add → new version searchable; the old version's ids never reappear
7. embedder failure mid-update → previous version fully queryable (I3)
8. Chroma failure during staging → previous version fully queryable (I3)
9. killed between stage and publish, restart, `reconcile` → consistent
10. `rebuild` restores dense search with **0** embed calls (I10)
11. a concurrent query during an update never returns mixed versions
12. duplicate chunks keep multiplicity; reorder-only edit → 0 embed calls

**Gate:** all 12 pass; `ingest_log` rows carry counts and timings but no text; clean clone
green.

### P7 — Retrieval, evidence gate, extractive answering — **DONE, see §8**

**7.1 Retrieval** (`retrieval/service.py`)

1. Normalise and tokenise the question with the same functions used for documents. Empty
   or whitespace → 422.
2. **Dense:** embed the query (LRU-cached) → `query(n = CANDIDATES_N × OVERFETCH)`.
3. **Lexical:** FTS5 `bm25` over `eligible_chunks`, top `CANDIDATES_N`.
4. **Eligibility:** join dense ids to `eligible_chunks` in one statement — which also
   applies `doc_ids` and `language` — fetching chunk text and metadata in that same
   statement. Anything not returned is dropped (I1, I2). If filtering leaves fewer than
   `CANDIDATES_N` while the vector store returned a full page, re-query once with a larger
   `n`, bounded.
5. **Fusion:** weighted RRF over the **union** of the two candidate lists —
   `score = w_d/(k+rank_d) + w_l/(rank_l)`. Keep raw dense similarity, `bm25`, and token
   coverage alongside the fused score for the gate and for debugging.
   > Fusing full-corpus dense ranks instead lets one rank-1 lexical hit outrank a correct
   > dense hit. That measures a system nobody will build, and it looks exactly like a
   > finding (D30). Fuse the candidate window.

**7.2 Evidence gate** (`retrieval/gate.py`) — runs **before** any answer text is chosen.

- Signals: `max_dense_similarity`, normalised lexical score, `token_coverage` (share of
  non-stopword query tokens present in the best chunk), dense/lexical top-k overlap,
  optional top-1/top-2 margin.
- Auditable rule, thresholds from `config/thresholds.json`, keyed by `model_id` so
  thresholds can never be reused across models:
  `passed = (max_dense ≥ min_dense AND coverage ≥ min_coverage) OR (lexical ≥ min_lexical AND coverage ≥ min_coverage_high)`
- Output is a structure, e.g.
  `{"passed": true, "reason": "dense+coverage", "max_dense": 0.83, "token_coverage": 0.61, "thresholds_version": 1, "calibrated": true}`,
  returned only when `debug=true`.
- Ship conservative placeholders flagged `"calibrated": false`. `/ready` reports the flag
  and the README must not claim calibration before P9 finishes.

**7.3 Extractive answerer** (`answering/extractive.py`)

- Split retrieved chunks into sentences, reusing the P3 splitter, offsets relative to the chunk.
- Score is IDF-weighted query-token overlap plus a small retrieval-score term. A sentence
  must meet `min_sentence_overlap` to be eligible — never pad with unrelated sentences from
  a relevant chunk.
- Select ≤ `MAX_ANSWER_SENTENCES`, order by source position, merge adjacent selections into
  one contiguous slice.
- Output **segments**: `{text, citation_id, chunk_char_start, chunk_char_end}` with
  `text == chunk.text[start:end]`. The `answer` string is a *rendering* (`text [n]`); `[n]`
  markers are system metadata, not content.
- **Never** paraphrase, translate, add connectives, or alter numbers, names, units, or
  negations. No world knowledge.
- **Insufficient evidence:** `status="insufficient_information"`, `citations=[]`, `answer` =
  a fixed localised system message (English: `Not enough information in the provided
  documents.`, with a Persian equivalent in a small message table), plus `reason` ∈
  `{empty_knowledge_base, no_relevant_content, below_threshold}`. The message is not
  evidence and is never cited. Never present "closest" chunks as normal citations.
- `evidence_score ∈ [0,1]` is a deterministic evidence score, **not** a probability. The
  name avoids `confidence` on purpose.

**7.4 Citation**

```json
{"id": 1, "doc_id": "handbook", "document": "handbook.md", "doc_version": 3,
 "section": "Guide > Install > Linux", "page": null, "lines": [42, 58],
 "chunk_id": "handbook:v3:7", "char_span": [120, 214], "excerpt": "exact source text",
 "score": 0.83}
```

Sentence→citation mapping is stable via the segment `citation_id`.

**Optional, only if P9 measures a gain:** dense re-scoring of candidate sentences behind
`SENTENCE_RERANK=true`, respecting the rate limiter.

**Tests first:** answerable → evidence from the expected version; unanswerable near-topic,
off-topic, empty KB → `insufficient_information` with no citations; **every segment is an
exact substring of its chunk and of `version.source_text` at the recorded offsets**; a
mutation that corrupts the answerer's output fails the substring test; deleted and outdated
content never appear in the answer, citations, or non-debug output; a Persian question
cites a Persian source with correct section, page, and line; gate threshold boundaries;
hostile FTS input stays inert.

**Gate:** all green; I1, I2, I6, I7 pass through the public retrieval-and-answer entry point.

**Status: done.** The plan's per-endpoint matrix was implemented as written, with these
deviations, each recorded:

* `POST /documents` returns **409** when the bytes differ from an existing active document and
  names `PUT` in the message. A POST that would silently replace content is ambiguous, and a
  caller who loses the version they thought they had is worse served than one told to be
  explicit.
* The service graph is built by `api/deps.py`, and **bringing the system up is `async`** —
  discovering a model's identity requires a network call, so there is no honest synchronous
  startup. `Services.aclose()` is async for the same reason: `httpx.AsyncClient` cannot be
  closed synchronously.
* `calibrate` and `eval` **report their missing precondition and exit 1** rather than
  half-existing. Both need `tests/eval/dataset.jsonl`, which is P9's §9.1. A calibration run
  against nothing would write numbers with no evidence behind them, which is the one thing
  `calibrated: false` exists to prevent.
* `/health` deliberately says nothing about dependencies. A liveness probe that fails on a
  database outage turns a recoverable problem into a restart loop.
* An **injected** service graph is not closed by the lifespan — the caller still holds it.

### P8 — API, observability, operations — **DONE, see §8**

| Method | Path | Behaviour |
|---|---|---|
| POST | `/documents` | multipart `file` + optional `doc_id` (default: slug of filename). `201` created · `200` unchanged · `409` exists with different content (use PUT) · `413` · `415` · `422` |
| PUT | `/documents/{doc_id}` | Replace an **active** document → `200` (`updated`/`unchanged`) · `404` unknown or deleted |
| DELETE | `/documents/{doc_id}` | `204` · `404` unknown; repeat delete → `404` |
| GET | `/documents` | Active documents, paginated (`limit`, `offset`) |
| GET | `/documents/{doc_id}` | Metadata, current version, section outline · `404` |
| POST | `/query` | `{question, top_k?, doc_ids?, language?, debug?}` → `answered` \| `insufficient_information` |
| GET | `/health` | Liveness, always `200` |
| GET | `/ready` | `200`/`503`: SQLite usable, FTS5 present, **local Chroma opens, lock held, dimension OK**, embedding model valid (cached, **no embedding call per request**), `thresholds_calibrated`, `embedder` provider |

```json
{"status": "answered", "answer": "Exact source excerpt [1].",
 "segments": [{"text": "Exact source excerpt.", "citation_id": 1}],
 "evidence_score": 0.82, "citations": [], "debug": null}
```

One error shape: `{"error": {"code": "DOCUMENT_NOT_FOUND", "message": "...", "request_id": "..."}}`.
Unknown exceptions → `500`, no stack traces, no secrets. A request-ID middleware logs ids,
counts, timings, and error classes only.

CLI (`python -m qasystem.cli`): `reconcile`, `rebuild`, `check-storage` (Chroma + SQLite
round-trip: upsert, query, delete, count), `calibrate`, `eval`.

**Tests first:** `TestClient` over every endpoint × happy path, validation path, domain
error, response shape; OpenAPI snapshot; upload size and content type; no secret in any
error body.

**Gate:** all green; starts with `--workers 1`; a second instance on the same `data/`
refuses to start (L2); `scripts/smoke_test.sh` passes: add → query → edit → query → delete
→ query.

### P9 — Evaluation and gate calibration

**9.1 Corpus and dataset — committed, deterministic, bilingual.** Author
`tests/eval/corpus/` from scratch: ~6 documents of 1–3 KB covering md, txt, pdf, in both
English and Persian. The 10 MB PDF is excluded as too large for a repeatable eval;
`ai-engineer.pdf` supplies a real Persian text layer.

`tests/eval/dataset.jsonl`, ≥50 questions: 15 answerable English · 10 answerable Persian ·
8 exact term/identifier/number · 5 multi-section · 8 unanswerable near-topic · 4
unanswerable off-topic.

Every answerable case uses **chunking-independent gold**: `{"doc": "handbook.md",
"must_contain": "exact phrase from source"}`. A hit is a retrieved chunk whose text contains
`must_contain`; chunk ids change whenever chunking is tuned, so they are never gold. Split
dev/test 60/40 **before** looking at results.

**9.2 Metrics**, for dense-only, lexical-only, and hybrid: R@1/3/5, MRR@5, unanswerable
false-answer rate, answerable refusal rate, citation substring validity (must be 100%),
stale-content leakage (must be 0), P50/P95 query latency, index size, ingest embed-request
count.

**9.3 Calibration.** Grid-search the gate on the **dev** split, maximising answerable recall
subject to unanswerable false-answer rate ≤5%, preferring 0% because the task is
evidence-first. Report the operating point on the held-out split. Write
`config/thresholds.json` with `model_id` (so a model change cannot silently reuse another
model's thresholds), `version`, `calibrated: true`, dataset size, and class balance. State
plainly that a dataset this small gives coarse estimates.

**9.4 Experiments, in this order**, adopting a change only if it improves dev **without**
hurting held-out:

1. **RRF weights.** Reproduce §2's table on the real dataset first; 0.7/0.3 measured worse
   than 0.9/0.1 at 12 questions, which is suggestive, not decisive.
2. **Chunk size** against BGE-M3's 40 949-char item ceiling — the largest measured cost
   lever (§2).
3. **Sentence reranking** behind `SENTENCE_RERANK=true`.
4. **The per-item cap**, currently 2.05× below the measured ceiling.

`scripts/measure_provider.py` caches fixture vectors on disk, so sweeps cost no API calls
after the first run.

**Gate:** `docs/eval_report.md` committed with real numbers for hybrid vs dense-only vs
lexical-only; README quotes them; thresholds calibrated and committed; invariants green.

### P10 — Hardening

- Run §11's checklist and close gaps. Coverage ≥85% on `text/`, `parsing/`, `ingestion/`,
  `retrieval/`, `answering/`.
- A regression test for every bug found along the way.
- A probe test for every §2/§3 behaviour production code depends on.

**Gate:** coverage targets met; every measured claim has a test that would fail if the third
party changed.

### P11 — Refactor and README

`jobTask.md`: "Code readability, separation of concerns, error handling, and testing key
components are important to us." This phase makes the work easy to review.

- **README** complete against the twelve items below, honest about limits. The model
  section already exists; extend it with the P9 numbers.

  1. Problem statement, constraints, assumption A1.
  2. Architecture diagram, and why there is **no** LLM generation: embedding-only service.
  3. **Embedding model: BGE-M3, and why** — latency, index size, multilingual and
     cross-lingual behaviour, measured retrieval quality, and the evidence's limits. The
     only place any other model is named. Links `docs/eval_report.md`.
  4. Storage design: SQLite as source of truth, FTS5 lexical, local persistent Chroma as a
     disposable derived index; data layout; backup by stopping and copying `data/`;
     `rebuild`.
  5. Version publication and deletion consistency: publish-then-cleanup, `eligible_chunks`.
  6. Chunking and Persian normalisation decisions.
  7. Retrieval, evidence-gate, and answer-selection methodology; what `evidence_score` means.
  8. Eval dataset, measured metrics, calibration procedure and its limits.
  9. API examples (`curl`) for upload, update, delete, query.
  10. Setup (`uv`, env vars, **single-worker requirement**) and test instructions.
  11. Known limitations: no OCR and unusable Persian text layers; extractive answers cannot
      synthesise across distant passages; single-process deployment; a small eval set; the
      84%-of-index fixture skew; PyMuPDF licence.
  12. Security: token externalised, redacted, never committed.
- **`docs/DECISIONS.md` as the reviewer's entry point**, ordered so the model rationale,
  thresholds, chunk size, and every deviation are findable in one read.
- **Invariant tests grouped** and runnable as `pytest -m invariant`, so I1–I10 are
  verifiable without reading the suite.
- **Dead code removed**; each module carries a one-line docstring stating role and layer.
  (`sqlite_store.py` is 559 lines against a ~400 target — splitting it needs a base class
  and a second connection, which is worse than the number; revisit here, D29.)
- **Layer boundaries verified** by extending the AST purity test past `domain/`.
- **Consistency pass:** error codes, config names, and docstrings agree with the OpenAPI
  schema and the README; no magic numbers inline.
- Verify a clean clone: `uv sync && cp .env.example .env && make check && make run`.

**Gate:** §11 fully checked.

---

## 10. Engineering rules

**Testing**

1. **Per phase:** the *Tests first* list is the contract. It must be red first.
2. **Mutation-check** each invariant and each safety property: break the code on purpose,
   name the test that fails, restore it. This is how D25 and D26 were found — a test that
   passes for the wrong reason is worse than no test, because it certifies the property.
3. **Fixture-wide coverage:** anything that should hold for a class of input is
   `parametrize`d over the class, so a new fixture is covered automatically.
4. **Storage tests use `tmp_path` only.** Never the real `data/`.
5. **The vector store is swappable** — a ~40-line in-memory fake satisfies `VectorStore`
   structurally, with no inheritance. Service tests use the fake, adapter tests real Chroma.
6. **Network is blocked** globally; live calls need `RUN_LIVE=1` and never print the token.
7. No test may depend on the repository being tidy — `test_repo_consistency.py` owns that.
8. Coverage is a signal, not a substitute for the invariants.

**Code**

9. Type hints everywhere; `mypy src` clean under `strict`. Small pure functions, explicit
   dependency injection, no global mutable state for clients, repositories, or config.
10. I/O behind ports and adapters. No bare `except:`. Convert low-level exceptions to typed
    domain errors at architectural boundaries.
11. `async` for network I/O; blocking SQLite and Chroma work in `asyncio.to_thread`.
12. Log identifiers, counts, timings, error classes — never document bodies or secrets.
13. Every tunable in config. No magic thresholds inline.
14. Every bug fix adds a regression test; every behaviour change updates README, OpenAPI,
    and tests in the same commit.
15. Never silently fall back to another model, another version, or a fabricated answer.
16. Do not add an abstraction until a second implementation or a test seam requires it.
17. **A deliberate simplification gets a `ponytail:` comment** naming the ceiling and the
    upgrade path, so the next reader knows it was a choice.
18. Prefer deleting. A file, branch, or option with no caller is a bug waiting to be
    documented as a feature.

## 11. Definition of Done

- [ ] PDF, TXT, Markdown ingestion works; English fixtures, Persian per A1
- [ ] Add / update / delete via API; identical re-upload → zero embedding requests
- [ ] Local edits re-embed only changed chunks; reorder-only edits re-embed nothing
- [ ] Deleted and superseded content provably unretrievable (I1–I3), cleanup fault-injected
- [ ] Failed updates never expose a partial version (I3)
- [ ] Chroma local and persistent; restart persistence verified; single-instance lock
      enforced; `rebuild` restores dense search with zero API calls (I10)
- [ ] Hybrid retrieval measured against dense-only and lexical-only baselines
- [ ] Answers are exact source excerpts; every citation mechanically verified (I6, I7)
- [ ] Unanswerable questions → `insufficient_information`, no citations
- [ ] No GUI dependency; stable OpenAPI; structured errors; no secrets exposed (I8)
- [ ] `make check` clean **and a fresh `git clone` green**
- [ ] `docs/eval_report.md`, `config/thresholds.json`, `docs/DECISIONS.md` committed;
      README complete and honest about limits

Reviewer questions, each answered by a named test:

| Question | Where it is answered |
|---|---|
| If Chroma still holds a deleted document's vectors, is it provably filtered out? | P6 test 5 (cleanup fault-injected) |
| Does a failed update leave the old version queryable? | P6 tests 7, 8 |
| Does an unchanged upload cost zero embed calls? A reorder-only edit? | P6 tests 2, 12 |
| Does a paragraph edit preserve most chunk hashes? Duplicates kept? | P3, P6 test 12 |
| After deleting `data/chroma/`, does `rebuild` restore search with no API calls? | P5 `rebuild` + tripwire test (D28) |
| Does a second process on the same `data/` fail fast? | P5 lock test, real subprocesses |
| Can a Persian query cite a Persian source with no manual RTL handling? | P7 `test_6_a_persian_question_cites_a_persian_source`, `test_6b_...pdf...` |
| Is every answer segment an exact substring of stored source text? | P7 `test_3_every_segment_is_an_exact_substring...` + M2/M13 mutations |
| Is user text ever interpreted as FTS5 syntax? | P5 hostile-input test |
| Is every answer segment an exact substring of stored source text? | P7 substring + mutation test |
| Do unanswerable questions avoid presenting "closest" evidence as citations? | P7 plan test 2 (6 cases) + M11 mutation |
| Is `evidence_score` documented as a non-probability? | README §7 |
| Can the model change without mixing dimensions, and are thresholds model-specific? | P5 I9 test; `thresholds.json` carries `model_id`; P7/P8 tests, and only *calibrated* numbers are refused across models (D42, D51) |
| Is the token absent from source control, logs, and error bodies? | P0/P4 redaction + scrub tests; P8 checks every response **and the access log**, which must record `url.path` and never the query string (D53) |
| Does a second instance on the same `data/` refuse to start? | P5 lock test; P8 starts a real second `uvicorn` and reads its log (L2) |
| Do the configured `SQLITE_PATH`/`CHROMA_PATH` actually take effect? | P8 `build_services` honours them; the bug where they were ignored is D52 |

## 12. Open questions for the human

1. **Fusion weights** — §2 measured 0.9/0.1 ahead of the 0.7/0.3 default, but at 12
   questions that is not decisive. Confirm P9's dev split should decide it rather than
   adopting the higher value now. `retrieve()` already accepts both weights so §9.2 can
   measure dense-only and lexical-only without touching config (D39).
1b. **Persian morphology** — coverage loses 0.29 to Ezafe suffixes (`خطایی` vs `خطا`), with
   no stemmer in place (D43). P9's Persian questions must include inflected terms or the eval
   will report morphology mismatch as retrieval quality. Decide whether a light Persian
   stemmer is in scope before §9.1 is authored.
2. **Chunk size** — raising `CHUNK_TARGET_TOKENS` toward 2 000-char chunks is worth ~2.9×
   throughput per character (9.5× at 8 000). Confirm it stays a P9 experiment, since it
   trades against retrieval precision.
3. **Cross-lingual retrieval versus the false-answer target — the one that needs a decision.**
   A live run measured that these cannot both hold at 5%: admitting an English question answered
   from a Persian source requires accepting uncorroborated dense hits, and one question whose answer
   is nowhere in the corpus (`قفل کردن نسخه`, dense 0.527) is admitted at the same similarity as
   the genuinely cross-lingual ones (0.519 / 0.528). **1 false answer in 9**, against §9.3's ≤5%
   target. Three options, and they are a product trade rather than a tuning one:
   **(a)** raise `min_dense_alone` — the false answer goes, cross-lingual goes with it;
   **(b)** keep it and accept a false-answer rate above target, documented;
   **(c)** gate uncorroborated hits on something other than cosine — a second lexical probe, or a
   language-aware signal.
   Recommended: **(c)**, with **(b)** as the interim. Also note the corpus is 84% one Persian book,
   so the measured bands are not balanced; P9's own corpus decides this properly.
4. **Live API during P9** — calibration and sweeps need `RUN_LIVE=1` and the real token.
   Everything else runs offline. 14 live tests now exist and pass.
