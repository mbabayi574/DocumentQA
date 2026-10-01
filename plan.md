# Document-Based QA System (LLM-free Extractive RAG) — v3

> **Audience:** the human reviewer of `jobTask.md`, and AI agents implementing it.
> **Version:** v3. Reorders v2 around `jobTask.md`, and folds in behaviour measured
> against the real provider and the real fixtures instead of assumed behaviour.
>
> **Core decision:** the only model service available is **embedding-only**
> (`GET /v1/models`, `POST /v1/embeddings`). There is no chat endpoint, so the system
> is deliberately **LLM-free and extractive**: retrieval finds evidence, the answerer
> returns **verbatim source text with citations**. Nothing may paraphrase, summarize,
> translate, or complete an answer.

## What changed from v2, and why

v2 was written before anything existed. v3 was written after probing the provider, the
SQLite build, Chroma 1.5.9, PyMuPDF, and every candidate fixture. Each change below
exists because a probe contradicted an assumption in v2.

| # | Change | Evidence |
|---|---|---|
| 1 | §1 is now `jobTask.md` → design → proof, in the reviewer's order. v2 led with internal architecture. | The grading criteria in `jobTask.md` are the six requirements; the plan should answer them first. |
| 2 | New §2 records **measured** provider/storage/fixture behaviour. | Six v2 assumptions were wrong; see §14. |
| 3 | Reassembly no longer trusts `index` alone. | Gemini returns `index: [0,0,0,0]`. |
| 4 | New `MAX_CHARS_PER_ITEM` config. | Bge-m3 rejects a single 45 000-char input: `maximum context length is 8192 token`. |
| 5 | `EmbeddingAuthError` covers `403 model-not-available`. | Provider returns 403 + allowed-ID list, not 404. |
| 6 | Retry never depends on `Retry-After`. | Header absent on 429. |
| 7 | Eval corpus is **authored and committed** (en + fa). | `tests/fixtures/docs/` was git-ignored and included a 10 MB 312-page PDF. |
| 8 | Persian PDF limitation stated, not worked around. | `justforfun_book_a4.pdf` extracts in lossy presentation forms; unrecoverable. |
| 8 | §1 row 3 now answers the model's graded requirement with the **rationale** the job asks for, backed by §2.2a. | `jobTask.md`: "choose the most appropriate one and explain your choice" — an explanation, not a bake-off. |
| 9 | New **P11 refactor pass**, since the job explicitly rewards presentation quality. | `jobTask.md`: "quality of the solution and the reasoning matter more than the number of tools". |
| 10 | `requires-python = ">=3.12"`. | Pinned `numpy==2.5.3` requires ≥3.12. |
| 11 | **One model: BGE-M3.** The four-way model comparison is dropped. §9.3 becomes a written rationale backed by §2.1 measurements plus one quality spot-check in §2.2a, and §9.4 spends the freed effort on tuning BGE-M3 instead. | Decided by the human. BGE-M3 is the fastest, the most compact, multilingual, and scored perfectly on the fixture spot-check, so a full four-way comparison buys nothing. |

---

## 0. Agent Operating Protocol

1. **Load minimal context.** Read this §0–§4 plus only the phase you are executing.
2. **One phase per session.** Work test-first: write the phase's *Tests first* list (red), implement (green), refactor.
3. **Gate before commit.** `make check` (`ruff check` + `ruff format --check` + `mypy src` + `pytest -q`) must be green and the phase's **Gate** items must pass.
4. **Never weaken tests to get green.** If a test looks wrong, log it in `docs/DECISIONS.md` and flag it.
5. **Verify third-party behaviour, don't recall it.** Every behaviour in §2 that code depends on gets a probe test.
6. **No real network in tests.** Live calls only under `RUN_LIVE=1`. A `socket.connect` guard in `tests/conftest.py` fails any test that tries.
7. **Secrets.** Read `EMBEDDING_API_KEY` from env only. Never commit, log, echo, snapshot, or return it.
8. **Scope guard — do NOT add:** OCR, auth, GUI, message queues, any LLM/chat call, other vector DBs, Chroma HTTP/server mode, silent model or version fallbacks.
9. **Stop-and-report.** After 3 failed attempts on one failing test: stop, summarize hypothesis and evidence, ask.
10. **Commit small**, one logical change each, tests included.
11. **Record decisions with evidence** in `docs/DECISIONS.md`.
12. **Keep chat output small.** Never dump embeddings, whole documents, or whole files into the conversation.

**Execution order.** Sequential, one commit per logical change. Lanes A–D from v2 remain logically independent but run in one session each, in dependency order, because the gates are cheap and P6 needs all four.

```text
P0 ✓ scaffold/contracts
  → P1 text → P2 parsing → P3 chunking → P4 embeddings → P5 storage
  → P6 ingestion [highest risk] → P7 retrieval+gate+answer → P8 API
  → P9 eval + BGE-M3 rationale + calibration → P10 hardening → P11 refactor + README
```

---

## 1. `jobTask.md` → design → proof

The six graded requirements, in the order `jobTask.md` states them.

| # | Requirement (verbatim intent) | Design answer | Proof |
|---|---|---|---|
| 1 | **Document processing: PDF, TEXT, Markdown, prepared for retrieval** | One parser per format → common `ParsedDocument` carrying the source text plus char/page/line offsets. Normalization + tokenizer feed FTS5; chunks feed the embedder. No text is ever rewritten in storage, so citations stay exact. | P2 fixture tests per format; P1 normalization tests; P3 slice-identity tests |
| 2 | **Change management: add/edit/delete; outdated content never used** | Versions are monotonic per `doc_id`. Publication is **one SQLite transaction** that flips `documents.current_version`. A single `eligible_chunks` view gates *every* retrieval path, so stale and deleted rows are unreachable even if Chroma cleanup never ran. Chroma is a derived, disposable index. | P6 tests 1–12: idempotency, edit isolation, delete-with-cleanup-fault-injected, failure injection, crash recovery |
| 3 | **Relevant retrieval** | Hybrid: dense (local Chroma, cosine) + lexical (FTS5 `bm25`), fused by weighted RRF. Candidate scores carry raw dense similarity, `bm25`, and token coverage for the gate. | P7 fusion/eligibility tests; P9 Recall@1/3/5 and MRR@5 for dense-only vs lexical-only vs hybrid |
| 4 | **Evidence-based, traceable answers** | Extractive answerer returns source slices only. Citations carry doc, version, section path, page, lines, chunk id, char span, and an exact excerpt. | P7 substring tests: every segment is an exact substring of its chunk and of `source_text` at the recorded offsets; mutation test proves the check bites |
| 5 | **Insufficient information: say so, never hallucinate** | An evidence gate runs **before** any answer text is selected. Below threshold → `status="insufficient_information"`, `citations=[]`, a fixed localized system message, and a machine-readable `reason`. "Closest" chunks are never shown as citations. | P7 unanswerable tests (near-topic, off-topic, empty KB); P9 false-answer rate target ≤5%, preference 0% |
| 6 | **API access, no GUI** | FastAPI + OpenAPI for upload/update/delete/query/health/ready; a CLI for maintenance. No GUI dependency anywhere. | P8 `TestClient` tests per endpoint; `scripts/smoke_test.sh` |
3. **Model selection — BGE-M3, justified by measurement.** All four candidates were measured for dimension, latency, and index cost; BGE-M3 was additionally spot-checked for retrieval quality, including cross-lingual English→Persian. The choice and its evidence limits are written up, not asserted. | §2.2a measurements; P9 `docs/eval_report.md` + `docs/DECISIONS.md`; README model section |
| 8 | **Readability, separation of concerns, error handling, tests** | `api → services → domain/ports ← adapters`. Typed errors with stable codes. Ports exist only where a second implementation or a test seam needs one. Invariants I1–I10 are release-blocking tests. | `make check`; coverage ≥85% on `text/ parsing/ chunking/ ingestion/ retrieval/ answering/`; P11 readability pass |

---

## 2. Measured environment (verified before writing any code)

Everything here was observed on this machine and this provider. **Code must not assume anything outside this section**, and each line here that code depends on gets a probe test.

### 2.1 Provider

| Observation | Value |
|---|---|
| `GET /v1/models` | 200. IDs exactly: `Bge-m3`, `Embedding-3-Small`, `Embedding-3-Large`, `Gemini-embedding-001` |
| Dimensions | `Bge-m3` **1024**, `Embedding-3-Small` **1536**, `Embedding-3-Large` **3072**, `Gemini-embedding-001` **3072** |
| Vector norms | Already L2-normalized server-side (‖v‖ = 1.0000) |
| `index` field | `Bge-m3` → `[0,1,2,3]`. **`Gemini-embedding-001` → `[0,0,0,0]`** — not a permutation |
| Response keys | Bge: `id, object, created, model, data, usage`. Others: `object, data, model, usage` |
| Rate limit | 130 rapid calls → exactly **120×200, 10×429** |
| `Retry-After` on 429 | **absent** |
| Request size | 225 000 chars → `413 input too large (225000 chars, max 200000)` |
| **Single-item size** | `Bge-m3` at 40 000 chars → 200; at 45 000 chars → `400 maximum context length is 8192 token` |
| Empty `input: []` | `400 please provide at least one prompt` |
| Unknown model | `403 model 'nope' is not available. Allowed: Bge-m3, Embedding-3-Large, Embedding-3-Small, Gemini-embedding-001` |
| Missing token | `401 missing or invalid token` |
| Batch size | 40 short items in one request → 200 |
| Latency, 2 items | Bge-m3 0.64s · E3-Small 3.80s · E3-Large 5.62s · Gemini 12.20s |
| Latency, 32 short items | Bge-m3 1.42s |
| `.env` line endings | CRLF. `pydantic-settings` strips them correctly; the token arrives clean |

### 2.2 Local stack

| Observation | Value |
|---|---|
| SQLite | 3.53.1, **FTS5 available** |
| Chroma | 1.5.9. `PersistentClient(path=..., settings=Settings(anonymized_telemetry=False))` works. Cosine via `metadata={"hnsw:space":"cosine"}`. Identical vector → distance `0.0`; orthogonal → `1.0`, so **similarity = 1 − distance**. `n_results` above `count()` is silently clamped, but we clamp explicitly anyway. Querying an empty collection returns empty lists rather than raising. `heartbeat()` returns a timestamp int |
| `filelock` | `FileLock(path, timeout=0)` in a second process raises `Timeout` as required by L2 |
| Python | 3.13 in `.venv`; `requires-python = ">=3.12"` because pinned `numpy==2.5.3` demands it |
| `make` | Was **absent** from this container; installed via `pacman`. `make check` now runs |

### 2.2a Why BGE-M3 — measured rationale

This is the evidence behind the single-model decision and the source material for the README section. `jobTask.md` says to choose the most appropriate model and explain why, so the explanation must be measured rather than asserted.

**Cost and speed** (2-item embed request, §2.1):

| Model | Dimension | Latency | vs BGE-M3 | Index cost per 1 000 chunks, float32 |
|---|---|---|---|---|
| **Bge-m3** | **1024** | **0.64s** | **1×** | **4.0 MB** |
| Embedding-3-Small | 1536 | 3.80s | 5.9× slower | 6.0 MB |
| Embedding-3-Large | 3072 | 5.62s | 8.8× slower | 12.0 MB |
| Gemini-embedding-001 | 3072 | 12.20s | 19.1× slower | 12.0 MB |

**Quality spot-check** on the real fixtures: 75 chunks from `storyen.md`, `storyfa.md`, and `ai-engineer.pdf`; 12 answerable questions (3 English, 5 Persian, 4 cross-lingual English→Persian) and 2 unanswerable. Pure dense retrieval, no chunking or fusion involved.

| Model | Recall@1 | Recall@3 | MRR@3 | Cross-lingual hits | Unanswerable top-1 wrong-doc |
|---|---|---|---|---|---|
| **Bge-m3** | **12/12** | **12/12** | **1.000** | **7/7** | 2/2 |

**Three reasons this holds up:**

1. **Multilingual by construction.** The corpus and the queries are bilingual and the cross-lingual queries retrieve Persian sources at 7/7 with no translation step, no language detector, and no per-language index. A1 makes bilingual behaviour a requirement, not a nice-to-have, and a single multilingual model removes an entire class of bugs (misrouted queries, language filters, per-language collections) that invariant I9 would otherwise have to police.
2. **Cheapest at equal quality.** Smallest dimension and lowest latency, with the best measured retrieval on the fixtures. There is no measured quality gain available from the larger models on this corpus, so their extra cost buys nothing here.
3. **Fits the quota.** At 120 req/min and 200 000 chars/request, ingest time is dominated by request count. BGE-M3's 6–19× lower latency keeps a re-ingest or a `rebuild` inside the rate limit comfortably, which matters because I10's `rebuild` re-reads the whole corpus from cache.

**Honest limits of this evidence**, to be stated in the README rather than hidden: one corpus, 75 chunks, 14 queries, no ablation over chunk size or fusion weights, and two unanswerable questions is too few to certify a false-answer rate. BGE-M3 also returned usable results at 40 000 chars per item but failed at 45 000 with an 8192-token context error (§2.1), so `MAX_CHARS_PER_ITEM` stays in place regardless of model. Claims about the other models are limited to the dimensions and latencies in §2.1 — no quality claim is made about models not benchmarked under the final configuration, and no pricing claim is made at all.

### 2.3 Fixtures (the corpus decision)

| File | Verdict |
|---|---|
| `en/storyen.md` (5.8 KB) | Usable English Markdown fixture |
| `fa/storyfa.md` (4.7 KB) | Usable Persian Markdown fixture, contains ZWNJ |
| `fa/ai-engineer.pdf` (8 pages) | **Usable Persian PDF.** Extracts correct Persian with `\u200c`. Page 1 has no text layer (blank), pages 2–7 carry the body |
| `en/Clean Code Fundamentals…pdf` (10 MB, 312 pages) | Text layer is real (347 893 chars, 1 blank page) but **too large to commit or to use in a repeatable eval**. Not part of the corpus |
| `fa/justforfun_book_a4.pdf` (204 pages) | **Unusable.** Extracts Persian in lossy presentation forms, character order reversed, ligatures decomposed to non-letters. No normalization recovers it. Excluded; documented as a known limitation |

**Corpus decision.** Stop depending on ad-hoc sample files. Author a small committed corpus under `tests/eval/corpus/`: ~6 documents across md/txt/pdf, **English and Persian both**, each 1–3 KB so a full eval run is fast and reproducible. Untrack the 10 MB PDF via `.gitignore`; track the small fixtures. The existing `storyen.md`/`storyfa.md`/`ai-engineer.pdf` stay as parser fixtures.

---

## 3. Constraints, assumptions, invariants

### 3.1 Constraints

| ID | Constraint | Consequence |
|---|---|---|
| C1 | Formats: PDF, TXT, Markdown | One parser per format; shared `ParsedDocument` |
| C2 | Edits/deletes must never leave stale evidence queryable | Version publication is the core consistency problem (P6) |
| C3 | Only `/v1/models` and `/v1/embeddings` exist | No generation. Extractive answers only |
| C4 | Candidates offered: BGE-M3, Embedding-3-Small, Embedding-3-Large, Gemini-Embedding-001 | **BGE-M3 is the model** (§2.2a). Still **never hard-code its ID or dimension**: discover the exact ID from `/v1/models` (§2.1: `Bge-m3`, not `BGE-M3`) and probe the dimension with one real call. The dimension is asserted against `EMBEDDING_DIMENSION` if set. **Never silently switch models** |
| C5 | 120 req/min, 200 000 chars/request | Client limiter ≤100/min. Batch on char budget **and** item cap **and** per-item cap (§2.1: 8192-token model context). Retry 429/5xx/timeouts with jittered backoff; `Retry-After` honored if present, absent in practice |
| C6 | Bearer token confidential | `SecretStr`, env-only, redaction filter |
| C7 | ChromaDB local persistent on-disk | `chromadb.PersistentClient`, in-process, one worker. See §4.2 |
| C8 | Graded on retrieval quality, substantiation, change management, readability, error handling, tests | Correctness and clarity over architectural breadth |

### 3.2 Assumption A1 — Persian support

`jobTask.md` does not mention languages; Persian/English is assumed from the provider's locale. English first, Persian as a bounded second-class workstream (~15% of effort), and **the P9 eval corpus is bilingual by decision**, so the choice of model is measured on both languages rather than assumed. Stated in the README.

**Known limitation (evidence, §2.3):** some Persian PDFs in the wild embed broken text layers that no parser can repair. We detect and report them rather than emit garbage. OCR is out of scope (§0.8).

### 3.3 Non-negotiable invariants (release blockers, each a test)

| ID | Invariant |
|---|---|
| I1 | **Active version:** every evidence item comes from a chunk whose `doc_version == documents.current_version` of an `active` document |
| I2 | **Deletion:** after the delete transaction commits, the document is unretrievable via lexical *and* dense paths **even if Chroma cleanup never ran** |
| I3 | **Atomic publication:** a failure before the publish transaction leaves the previous version fully queryable and unchanged |
| I4 | **Idempotency:** re-uploading identical bytes — or bytes yielding identical parsed text — returns `unchanged` with **zero** embedding requests and no new version |
| I5 | **Reuse:** unchanged chunks cause zero embedding requests (cache keyed by the exact embedded input) |
| I6 | **Traceability:** `source_text[char_start:char_end] == chunk.text`, and every answer segment is an exact substring of its cited chunk |
| I7 | **No synthesis:** answer text is selected source slices in source order, with `[n]` markers rendered outside the slices |
| I8 | **Secrets:** the token never appears in logs, errors, snapshots, fixtures, or committed files |
| I9 | **Model isolation:** one Chroma collection per `(model_id, dimension)`; vectors of different models or dimensions never mix |
| I10 | **Disposable index:** deleting `data/chroma/` and running `rebuild` restores dense search from SQLite with **zero** embedding API calls |

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

### 4.1 Key design decisions

- **SQLite is the single source of truth**: documents, versions, chunk text and offsets, FTS5, embedding cache, ingest log. Publication state lives only here.
- **Chroma is a derived, rebuildable index** holding vectors and diagnostic metadata only, never document text. Because `embedding_cache` keeps every vector, `rebuild` needs no API calls (I10).
- **Publish = one SQLite transaction** flipping `documents.current_version`. No second "is active" flag; eligibility is derived through a view (§6).
- **Versions are monotonic per `doc_id`, even across delete/re-add.** Vector IDs are `"{doc_id}:v{version}:{ordinal}"`, so a stale vector can never collide with a fresh one.
- **Publish, then clean up.** Embed and stage first, publish atomically, purge superseded rows and vectors best-effort after. An incomplete update is ignorable; a published update is complete.
- No message queue: the API shape does not need one.

### 4.2 Local Chroma rules

| # | Rule |
|---|---|
| L1 | `chromadb.PersistentClient(path=settings.chroma_path, settings=Settings(anonymized_telemetry=False))`. **Never** `HttpClient`, `EphemeralClient`, or an in-memory fallback outside tests |
| L2 | **Single-owner process.** `uvicorn --workers 1`. At startup take an exclusive `filelock.FileLock(data/.qasystem.lock, timeout=0)`; if held, exit with a clear error. Verified available (§2.2) |
| L3 | Chroma calls are blocking: wrap in `asyncio.to_thread`. Serialize Chroma + SQLite writes behind one application write lock |
| L4 | Collection name `chunks__{slug(model_id)}__d{dimension}` (`[a-z0-9_-]`, <63 chars). Store `model_id` and `dimension` in metadata and **assert on open** (I9) |
| L5 | Cosine via `metadata={"hnsw:space":"cosine"}` on the **pinned 1.5.9**, proven by a known-vector test (§2.2). `similarity = 1 − distance` |
| L6 | Store **no document text** in Chroma. Metadata: `doc_id`, `doc_version`, `ordinal`. SQLite filtering decides eligibility |
| L7 | Clamp `n_results` to `count()` and skip the query when the count is 0. Over-fetch `CANDIDATES_N × OVERFETCH` so filtered-out stale vectors do not starve the candidate list |
| L8 | `data/` is git-ignored. **Backup = stop the app, copy `data/`.** Never mutate Chroma's internal files directly |
| L9 | Readiness: data dir writable, lock held, `heartbeat()`, collection opens, dimension matches config. Report failures explicitly; never silently recreate a different store |
| L10 | Tests use a real `PersistentClient` on `tmp_path`. A **restart test** writes → drops the client → reopens the same path → queries → still found |

---

## 5. Contracts (P0, frozen)

In `src/qasystem/domain/`. The domain layer must not import FastAPI, Chroma, httpx, or sqlite3 — enforced by an AST test.

```python
# domain/models.py
@dataclass(frozen=True)
class Section:
    char_start: int; char_end: int             # slice of ParsedDocument.text
    section_path: tuple[str, ...]              # ("Guide", "Install", "Linux")
    page_start: int | None; page_end: int | None
    line_start: int | None; line_end: int | None

@dataclass(frozen=True)
class Chunk:
    ordinal: int
    char_start: int; char_end: int
    text: str
    section_path: tuple[str, ...]
    page_start: int | None; page_end: int | None
    line_start: int | None; line_end: int | None
    chunk_hash: str                            # sha256(canonical(section_path) + normalized text)
    language: Literal["fa", "en", "mixed"]

@dataclass(frozen=True)
class ParsedDocument:
    title: str
    text: str                                  # "source text": all offsets refer to this
    sections: tuple[Section, ...]
    format: Literal["pdf", "txt", "md"]

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

Application services, not ports: `IngestionService`, `RetrievalService`, `AnswerService`. **A new Protocol is added only when a second implementation or a test seam requires it.**

Typed errors (`errors.py`), each with a stable `code` and HTTP mapping:

| Error | Code | HTTP |
|---|---|---|
| `UnsupportedFormatError` | `UNSUPPORTED_FORMAT` | 415 |
| `FileTooLargeError` | `FILE_TOO_LARGE` | 413 |
| `EmptyDocumentError` / `ParseError` / `NoTextLayerError` | `EMPTY_DOCUMENT` / `PARSE_ERROR` / `NO_TEXT_LAYER` | 422 |
| `DocumentNotFoundError` | `DOCUMENT_NOT_FOUND` | 404 |
| `DocumentExistsError` (POST with existing active `doc_id`, different content) | `DOCUMENT_EXISTS` | 409 |
| `EmbeddingUnavailableError` (retries exhausted) | `EMBEDDING_UNAVAILABLE` | 503 |
| `EmbeddingAuthError` (401/403, incl. model-not-available per §2.1) | `EMBEDDING_AUTH` | 502 |
| `VectorStoreError` | `VECTOR_STORE_UNAVAILABLE` | 503 |
| `ConfigError` | `CONFIG_ERROR` | 500 |
| `InsufficientEvidence` is **not** an error — a normal `200` | — | 200 |
| anything else | `INTERNAL_ERROR` | 500, no stack trace, no secrets |

---

## 6. Data model (SQLite)

Pragmas: `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout`. Short-lived connections, one write lock.

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
  chunk_id TEXT PRIMARY KEY,              -- "{doc_id}:v{version}:{ordinal}" == Chroma vector id
  doc_id, doc_version, ordinal, text TEXT NOT NULL,
  char_start INT, char_end INT,
  chunk_hash TEXT, embed_input_hash TEXT, -- hash of the exact string that was embedded
  section_path_json TEXT, page_start, page_end, line_start, line_end, language,
  UNIQUE(doc_id, doc_version, ordinal))

chunks_fts  -- FTS5(chunk_id UNINDEXED, tokens); tokens = OUR tokenizer's output, space-joined
            -- rows for staging AND published versions; eligibility applied by join

embedding_cache(
  input_hash TEXT, model_id TEXT, dim INT, vector BLOB /* float32 */, created_at,
  PRIMARY KEY(input_hash, model_id))
  -- key = sha256 of the exact embedded string, so any change to the breadcrumb or
  -- template invalidates correctly; no separate "input version" column needed.

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

Lexical search stores **our** tokenizer's output (including ZWNJ split components) in FTS5, so SQLite's tokenizer never needs to understand Persian. Query with `bm25()` joined to `eligible_chunks`. User text is tokenized and each token is emitted as a **quoted** FTS term joined by `OR` — never passed as raw FTS5 syntax. Startup check: FTS5 is present in this Python's SQLite build.

---

## 7. Configuration (`config.py`, validated at startup)

```dotenv
APP_ENV=dev                          # dev | test | demo | prod
SENTENCE_RERANK=false                # optional P7/P9 experiment

# --- embedding service ---
EMBEDDING_BASE_URL=https://models-interview.arvancloudai.ir/v1
EMBEDDING_API_KEY=<secret, required when provider=remote>
EMBEDDING_MODEL=<exact id from GET /v1/models>   # §2.1: Bge-m3, not BGE-M3
EMBEDDING_PROVIDER=remote            # remote | fake (fake only in APP_ENV test/demo; shown in /ready)
EMBEDDING_DIMENSION=                 # optional; if set, must equal the probed dimension (BGE-M3: 1024)

# --- storage (all local) ---
DATA_DIR=./data
SQLITE_PATH=./data/qasystem.db
CHROMA_PATH=./data/chroma
MAX_UPLOAD_MB=20

# --- chunking ---
CHUNK_TARGET_TOKENS=350
CHUNK_HARD_MAX_TOKENS=700
CHUNK_OVERLAP_RATIO=0.15

# --- embedding client ---
MAX_CHARS_PER_REQUEST=160000         # §2.1: provider cap is 200000
MAX_CHARS_PER_ITEM=20000             # NEW: BGE-M3 context is 8192 tokens; 45k chars fails, 40k passes
MAX_ITEMS_PER_BATCH=32
RATE_LIMIT_PER_MIN=100               # measured ceiling is 120 (§2.1)
REQUEST_TIMEOUT_S=30
MAX_RETRIES=5

# --- retrieval ---
CANDIDATES_N=30
OVERFETCH=2
TOP_K=5
RRF_K=60
DENSE_WEIGHT=0.7
LEXICAL_WEIGHT=0.3

# --- gate + answer (written by `make calibrate`) ---
THRESHOLDS_PATH=./config/thresholds.json
MAX_ANSWER_SENTENCES=5
```

Every tunable lives here, never inline. `.env` is git-ignored; commit only `.env.example` with placeholders.

---

## 8. Repository layout

```text
├── AGENTS.md  README.md  plan.md  Makefile  pyproject.toml  .env.example  .gitignore
├── config/thresholds.json            # produced by `make calibrate`, committed
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
├── tests/ unit/ integration/ api/ eval/{dataset.jsonl,runner.py,corpus/} fixtures/docs/{en,fa}/
├── data/                               # git-ignored: qasystem.db, chroma/, .qasystem.lock
└── scripts/ smoke_test.sh  eval.sh
```

Dependency direction: `api → services → domain/ports ← adapters (parsing, storage, embeddings)`.

---

## 9. Phases

Each phase lists **Build**, **Tests first**, **Gate**. Do not start a dependent phase until the Gate passes.

---

### P0 — Scaffold, configuration, contracts `[S]` — **DONE**

Built: `uv` project with pinned deps (D2/D3/D4/D5 in `docs/DECISIONS.md`), `Makefile` with all seven targets, `.env.example`, `.gitignore`, `config.py` (`SecretStr`, provider-cap validation), `errors.py`, `logging_setup.py` (redaction filter), `domain/models.py`, `domain/ports.py`, `embeddings/fake.py`, app factory with `GET /health`.

**Gate — verified:** `make check` green (61 tests); boots with `EMBEDDING_PROVIDER=fake APP_ENV=test` under `uvicorn --workers 1`; no test can open a socket (enforced in `tests/conftest.py`, with a test proving the guard); token absent from all tracked files.

---

### P1 — Text layer: normalization, tokenization, language `[S]`

**Build**
- `normalize_for_index(s)`: NFKC; Arabic `ي/ك → ی/ک`; Arabic-Indic and Persian digits → ASCII; strip tatweel and diacritics; collapse whitespace; **keep and normalize ZWNJ** (U+200C), never blind-delete; drop other control and zero-width characters. **Stored and cited text is never normalized** — only index and query strings.
- `tokenize(s)`: lowercase Latin; keep ZWNJ compounds **and** emit split components for recall; preserve identifiers (`ERR-404`, `v2.3.1`); no stemming unless P9 measures a gain. Small built-in stopword sets (en/fa) used only for coverage and sentence scoring, never for FTS.
- `detect_language(s) -> fa|en|mixed` by script ratio.

**Tests first:** Persian variants match; queries with and without ZWNJ hit the same fixture; idempotence `f(f(x)) == f(x)` under Hypothesis; mixed fa/en and numeric identifiers; English unchanged apart from case and whitespace.

**Gate:** all green; functions pure, no I/O.

---

### P2 — Parsing: Markdown, TXT, PDF `[M]`

Every parser returns a `ParsedDocument` whose offsets index its own `text`.

- **Markdown** — `markdown-it-py` token line maps, not regexes. ATX and Setext headings; `#` inside fenced code ignored; breadcrumb `Guide > Install > Linux`; text before the first heading takes the document title as root; line spans converted to char offsets.
- **TXT** — decode UTF-8-SIG then UTF-8; a Windows-1256 fallback only if a fixture justifies it, otherwise a typed decode error; split on blank lines; keep line spans; normalize only `\r\n → \n`.
- **PDF (PyMuPDF)** — page by page, `get_text("blocks", sort=True)`; exact 1-based page numbers; headings only when font-size heuristics are reliable, otherwise page-based sections; **never reverse RTL strings unless a fixture proves the parser output needs it**. Encrypted, corrupt, or text-layer-free PDFs raise `NoTextLayerError` — there is no OCR, and we say so in the message.
- **Registry** — choose by extension plus PDF magic bytes (`%PDF-`); enforce `MAX_UPLOAD_MB` **before** parsing; unsupported → `UnsupportedFormatError`; empty or whitespace-only → `EmptyDocumentError`.

**Tests first:** English and Persian fixtures for MD and TXT; breadcrumbs, line spans, page numbers correct; **blank-page handling proven against `ai-engineer.pdf` page 1** (§2.3) so one empty page does not void a document; corrupt PDF; scanned PDF; empty file; oversize file; wrong magic bytes; `justforfun_book_a4.pdf` behaviour documented as a known limitation rather than silently mangled.

**Gate:** fixtures pass; `text[s.char_start:s.char_end]` non-empty for every section.

---

### P3 — Deterministic chunking `[M]`

- Heading-aware: never merge unrelated top-level sections; merge small **adjacent sibling** sections only if the result stays within target; split oversized sections with sentence-aware windows.
- Target ≈ `CHUNK_TARGET_TOKENS`, overlap ≈ `CHUNK_OVERLAP_RATIO`, hard cap `CHUNK_HARD_MAX_TOKENS`. A documented chars-per-token heuristic by script, not a fake exact tokenizer.
- Sentence boundaries `. ! ? ؟ ۔`, avoiding decimals and abbreviations (`3.14`, `e.g.`, `Dr.`).
- **Every chunk is a contiguous slice** of `ParsedDocument.text`; overlap is an earlier start offset. This is what makes I6 mechanically checkable.
- `chunk_hash = sha256(canonical(section_path) + normalize_for_index(text))`.
- Embedding input, built in the ingestion layer and never stored in Chroma: `"{title} > {section breadcrumb}\n\n{raw chunk text}"`. Citations always use the raw slice.

**Tests first:** determinism, byte-identical on re-run; every source paragraph covered by ≥1 chunk; no chunk over the hard cap; slice identity `text == source[char_start:char_end]`; a local paragraph edit preserves ≥80% of unaffected `chunk_hash`es (Hypothesis plus fixtures).

**Gate:** all green.

---

### P4 — Embeddings client, batching, cache `[M]`

- Async `httpx`: `GET /models`, `POST /embeddings` with `{"model": ..., "input": [...]}`.
- **Discovery at startup:** the configured model must exactly match an ID from `/models`; the error lists available IDs and **never** the token. **Probe** the dimension with one embedding call; if `EMBEDDING_DIMENSION` is set it must match.
- **Reassembly:** accept `index` only when it is a permutation of `0..n-1`. Otherwise fall back to response order and log a warning. *Required by §2.1: Gemini returns `[0,0,0,0]`.* Every vector's dimension is validated.
- **Batching** on three bounds: per-request character budget, item cap, and **per-item** character cap (`MAX_CHARS_PER_ITEM`, §2.1's 8192-token model context).
- Token-bucket limiter at ≤`RATE_LIMIT_PER_MIN`.
- Retry only 429, 5xx, connection resets, timeouts. Honor `Retry-After` when present — §2.1 measured it absent — otherwise capped exponential backoff with jitter. 401 and 403 fail fast as `EmbeddingAuthError`, the latter carrying the allowed-model list from the provider message. Retries exhausted → `EmbeddingUnavailableError`.
- `CachingEmbedder(inner, cache_repo)`: dedupe inputs within a call, look up `(sha256(input), model_id)`, embed only misses, write back. Small in-memory LRU for query embeddings.

**Tests first (respx, no real network):** order reassembly for a shuffled `index`; fallback when `index` is not a permutation, modeled on the observed Gemini response; batches never exceed any of the three bounds; 429 with and without `Retry-After`; persistent 5xx → typed error; 401/403 no retry; dimension mismatch; model ID absent from `/models`; duplicates in one call cause one embed; a cache hit causes zero requests.

**Optional live test:** only under `RUN_LIVE=1`; never prints the token.

**Gate:** all green; the limiter test proves ≤ configured rate under a burst using a fake clock.

---

### P5 — Storage layer (SQLite + FTS5 + local Chroma) `[M]`

- `schema.sql` per §6, idempotent creation, `PRAGMA user_version` for migrations; repository methods with one obvious transaction boundary per operation.
- `lexical.py`: token-quoted `OR` queries, `bm25()` rank, normalized lexical score, **always joined to `eligible_chunks`**.
- `chroma_store.py`: `VectorStore` over `PersistentClient` per L1–L10. Deterministic collection name, cosine space, `similarity = 1 − distance`, explicit `n` clamping, paged `list_ids`, `ping`.
- `ingestion/reconcile.py` primitives used by P6 and P10: `plan_reconcile()` compares SQLite's expected vector IDs with `list_ids()`; `rebuild()` re-upserts from `embedding_cache` with no API calls.

**Tests first:** schema idempotent; transaction rollback; `eligible_chunks` excludes staged, superseded, and deleted rows; FTS with hostile input `" OR * ( ) NEAR`; Chroma cosine conversion on known vectors; **restart persistence** (L10); collection and dimension mismatch rejected (I9); `n` clamping on tiny collections; a second process cannot take the lock (L2).

**Gate:** storage tests run on `tmp_path` only; the Chroma adapter is swappable for a fake in service tests.

---

### P6 — Ingestion and change management `[L]` — **highest risk, before any retrieval tuning**

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
2. The version number is allocated from `documents.last_version + 1` inside a transaction, never reused.
3. Diff by `chunk_hash` with **multiset** matching (hash → queue) so duplicates keep multiplicity; a reorder alone triggers zero embed calls. "Reused" means the **embedding** is reused from cache and re-upserted under the new versioned ID; local Chroma upserts are free.
4. Any failure before publish → version `failed`, staged rows and vectors removed best-effort, previous version untouched (I3). Stragglers swept by `reconcile`.
5. **Delete:** one transaction sets `status='deleted'`, `current_version=NULL`; from that commit the document is unretrievable (I2). Purge chunks, FTS rows, and vectors best-effort. Keep the `documents` tombstone so `last_version` stays monotonic.
6. **Re-add** after delete creates version `last_version+1`.
7. If Chroma staging fails, **abort**. No lexical-only degraded publish.
8. Startup `reconcile`: fail stale `staging` versions; report or remove Chroma IDs SQLite does not expect; **never** mutate SQLite based on Chroma's contents; if the collection is missing or short, run `rebuild`.
9. The write lock serializes ingestions. A query reads one SQLite snapshot — candidates joined to `eligible_chunks` returning chunk text in one statement — so a query never sees two versions of one document.

**Tests first** — `FakeEmbedder` + real SQLite + real `PersistentClient` on `tmp_path`:

1. add → query finds content
2. identical re-upload → `unchanged`, **0** embed calls; same parsed text from different bytes → also 0
3. local edit → only changed chunks embedded
4. after publish, old text is unretrievable
5. delete → unretrievable lexically **and** densely, **with Chroma cleanup fault-injected to fail** (I2)
6. re-add → new version searchable; the old version's IDs never reappear
7. embedder failure mid-update → previous version fully queryable (I3)
8. Chroma failure during staging → previous version fully queryable (I3)
9. kill between stage and publish, restart, `reconcile` → consistent
10. delete `data/chroma/`, run `rebuild` → dense search restored, **0** embed calls (I10)
11. a concurrent query during an update never returns mixed versions
12. duplicate chunks keep multiplicity; reorder-only edit → 0 embed calls

**Gate:** all 12 pass; `ingest_log` rows carry counts and timings but no text.

---

### P7 — Retrieval, evidence gate, extractive answering `[L]`

**7.1 Retrieval** (`retrieval/service.py`)
1. Normalize and tokenize the question with the same functions used for documents. Empty or whitespace → 422.
2. **Dense:** embed the query (LRU-cached) → `VectorStore.query(n = CANDIDATES_N × OVERFETCH)`.
3. **Lexical:** FTS5 `bm25` over `eligible_chunks`, top `CANDIDATES_N`.
4. **Eligibility:** join dense IDs to `eligible_chunks` in one SQL statement — which also applies `doc_ids` and `language` filters — fetching chunk text and metadata in that same statement. Anything not returned is dropped (I1, I2). If filtering leaves fewer than `CANDIDATES_N` while Chroma returned a full page, re-query once with a larger `n`, bounded.
5. **Fusion:** weighted RRF `score = w_d/(k+rank_d) + w_l/(k+rank_l)`, keeping raw dense similarity, `bm25`, and token coverage alongside the fused score for the gate and for debugging. Treat 0.7/0.3 as a baseline to evaluate, not a truth.

**7.2 Evidence gate** (`retrieval/gate.py`) — runs **before** any answer text is chosen.
- Signals: `max_dense_similarity`, normalized lexical score, `token_coverage` (share of non-stopword query tokens present in the best chunk), dense/lexical top-k overlap, optional top-1/top-2 margin.
- Rule, auditable, thresholds from `config/thresholds.json`, **keyed by `model_id`** because similarity distributions differ per model (§2.1 shows dimensions and latency differ by 2× and 19×):
  `passed = (max_dense ≥ min_dense AND coverage ≥ min_coverage) OR (lexical ≥ min_lexical AND coverage ≥ min_coverage_high)`
- Output is a structure, e.g. `{"passed": true, "reason": "dense+coverage", "max_dense": 0.83, "token_coverage": 0.61, "thresholds_version": 1, "calibrated": true}`, returned only when `debug=true`.
- Ship conservative placeholders flagged `"calibrated": false` until P9. `/ready` reports the flag and the README must not claim calibration before P9 finishes.

**7.3 Extractive answerer** (`answering/extractive.py`)
- Split retrieved chunks into sentences, reusing the P3 splitter, keeping offsets relative to the chunk.
- Score is IDF-weighted query-token overlap plus a small retrieval-score term. A sentence must meet `min_sentence_overlap` to be eligible — never pad with unrelated sentences from a relevant chunk.
- Select ≤ `MAX_ANSWER_SENTENCES`, order by source position, merge adjacent selections into one contiguous slice.
- Output **segments**: `{text, citation_id, chunk_char_start, chunk_char_end}` with `text == chunk.text[start:end]`. The `answer` string is a *rendering* (`text [n]`); `[n]` markers are system metadata, not content.
- **Never** paraphrase, translate, add connectives, alter numbers, names, units, or negations, or use world knowledge.
- **Insufficient evidence:** `status="insufficient_information"`, `citations=[]`, `answer` = a fixed localized system message (English: `Not enough information in the provided documents.`, with a Persian equivalent in a small message table), plus `reason` ∈ `{empty_knowledge_base, no_relevant_content, below_threshold}`. The message is not evidence and is never cited. Never present "closest" chunks as normal citations.
- `evidence_score ∈ [0,1]` is a deterministic evidence score, **not** a probability. The field name avoids `confidence` on purpose.

**7.4 Citation model**
```json
{"id": 1, "doc_id": "handbook", "document": "handbook.md", "doc_version": 3,
 "section": "Guide > Install > Linux", "page": null, "lines": [42, 58],
 "chunk_id": "handbook:v3:7", "char_span": [120, 214],
 "excerpt": "exact source text", "score": 0.83}
```
Sentence→citation mapping is stable via the segment `citation_id`.

**Optional, only if P9 measures a gain:** dense re-scoring of candidate sentences, behind `SENTENCE_RERANK=true`, respecting the rate limiter.

**Tests first:** answerable → evidence from the expected version; unanswerable near-topic, off-topic, and empty-KB → `insufficient_information` with no citations; **every segment is an exact substring of its chunk and of `version.source_text` at the recorded offsets**; a mutation test that corrupts the answerer's output fails the substring test; deleted and outdated content never appear in answer, citations, or non-debug output; a Persian question cites a Persian source with correct section, page, and line; gate threshold boundary cases; hostile FTS input stays safe.

**Gate:** all green; I1, I2, I6, I7 tests pass through the public retrieval and answer entry point.

---

### P8 — API, observability, operations `[M]`

| Method | Path | Behavior |
|---|---|---|
| POST | `/documents` | multipart `file` plus optional `doc_id` (default: slug of filename). `201` created · `200` unchanged · `409` exists with different content (use PUT) · `413` · `415` · `422` |
| PUT | `/documents/{doc_id}` | Replace an **active** document → `200` (`updated` or `unchanged`) · `404` unknown or deleted |
| DELETE | `/documents/{doc_id}` | `204` · `404` unknown; repeat delete → `404` |
| GET | `/documents` | Active documents, paginated (`limit`, `offset`) |
| GET | `/documents/{doc_id}` | Metadata, current version, section outline · `404` |
| POST | `/query` | `{question, top_k?, doc_ids?, language?, debug?}` → `answered` \| `insufficient_information` |
| GET | `/health` | Liveness, always `200` |
| GET | `/ready` | `200`/`503`: SQLite usable, FTS5 present, **local Chroma opens, lock held, dimension OK**, embedding model valid (cached, **no embedding call per request**), `thresholds_calibrated`, `embedder` provider |

Query response:
```json
{"status": "answered",
 "answer": "Exact source excerpt [1].",
 "segments": [{"text": "Exact source excerpt.", "citation_id": 1}],
 "evidence_score": 0.82,
 "citations": [ /* §7.4 */ ],
 "debug": null}
```

Error shape, one stable form: `{"error": {"code": "DOCUMENT_NOT_FOUND", "message": "...", "request_id": "..."}}`. Unknown exceptions → `500` with no stack traces and no secrets. A request-ID middleware logs ids, counts, timings, and error classes only.

CLI (`python -m qasystem.cli`): `reconcile`, `rebuild`, `check-storage` (Chroma plus SQLite round-trip: upsert, query, delete, count), `calibrate`, `eval`.

**Tests first:** FastAPI `TestClient` over every endpoint × happy path, validation path, domain error, and response shape; OpenAPI schema snapshot; upload-size and content-type handling; no secret in any error body.

**Gate:** all green; the server starts with `--workers 1`; a second instance on the same `data/` refuses to start (L2); `scripts/smoke_test.sh` passes: add → query → edit → query → delete → query.

---

### P9 — Evaluation, BGE-M3 rationale, gate calibration `[M]`

**9.1 Corpus and dataset — committed, deterministic, bilingual.**

Author `tests/eval/corpus/` from scratch: ~6 documents of 1–3 KB each, covering md, txt, and pdf, in **both English and Persian**. The 10 MB sample PDF is excluded as too large for a repeatable eval (§2.3); `ai-engineer.pdf` supplies a real Persian PDF text layer.

Then `tests/eval/dataset.jsonl`, ≥50 questions:

| Type | Target |
|---|---|
| Answerable, English | 15 |
| Answerable, Persian (A1) | 10 |
| Exact term / identifier / number | 8 |
| Multi-section | 5 |
| **Unanswerable** — near-topic vocabulary | 8 |
| **Unanswerable** — off-topic | 4 |

Every answerable case uses a **chunking-independent gold**: `{"doc": "handbook.md", "must_contain": "exact phrase from source"}`. A retrieval hit is a retrieved chunk whose text contains `must_contain`; chunk IDs change whenever chunking is tuned, so they are never gold. Split dev/test 60/40 **before** looking at results.

**9.2 Metrics**, reported for dense-only, lexical-only, and hybrid: Recall@1/3/5, MRR@5, unanswerable false-answer rate, answerable refusal rate, citation substring validity (must be 100%), stale-content leakage (must be 0), P50/P95 query latency, index size, ingestion embed-request count.

**9.3 Model rationale.** No comparison phase. BGE-M3 is the model (§2.2a carries the measurements). What this phase owes the reviewer is a written justification in `docs/DECISIONS.md` and the README, citing: measured dimension and latency for all four candidates, the multilingual cross-lingual result, the index-size arithmetic, and the rate-limit argument. State the evidence's limits plainly (§2.2a): one small corpus, no ablation, few unanswerable cases.

**What we do instead of comparing models — exploit BGE-M3 properly.** Its measured properties justify specific implementation choices, and each is a test, not a claim:

| BGE-M3 property | How the system uses it |
|---|---|
| 8192-token context, 40 000 chars OK, 45 000 fails (§2.1) | `MAX_CHARS_PER_ITEM=20000` leaves 2× headroom. Chunks are sized for this window, so one chunk is always one embed call |
| Vectors arrive L2-normalized (‖v‖=1.0) | Skip normalizing on ingest; assert ‖v‖≈1 in a probe test so a provider change is caught rather than silently skewing cosine |
| Dense multilingual retrieval, 7/7 cross-lingual | No language routing, no translation, no per-language collection. One collection for both languages (§2.2a reason 1) |
| Lowest latency and smallest dimension | Fast `rebuild` (I10) and re-ingest stay well inside the 120 req/min quota (§2.2a reason 3) |
| Strong lexical component in the same vector | Does **not** replace FTS5. Lexical retrieval still runs because `must_contain` gold phrases and error codes like `ERR-404` need exact matching that dense similarity does not guarantee |

**9.3b Threshold calibration.** `make calibrate` grid-searches the gate on the **dev** split, maximizing answerable recall subject to unanswerable false-answer rate ≤5%, preferring 0% because the task is evidence-first. Report the operating point on the held-out split. Write `config/thresholds.json` with `model_id` (so switching models later cannot silently reuse another model's thresholds), `version`, `calibrated: true`, dataset size, and class balance. State plainly that a dataset this small gives coarse estimates.

**9.4 Optimization experiments, BGE-M3 only.** Worth running, in this order, adopting a change only if it improves the dev metric without hurting held-out results: RRF weights (0.7/0.3 is a starting point, not a truth), chunk size against BGE-M3's context window, sentence reranking behind `SENTENCE_RERANK=true`, and the per-item cap. This is where effort goes now that the model comparison is gone.

**Gate:** `docs/eval_report.md` committed with real numbers for hybrid versus dense-only versus lexical-only; the README quotes them alongside the §2.2a model rationale; thresholds calibrated and committed; invariants still green.

---

### P10 — Hardening `[S]`

- Run the final checklist (§13) and close gaps. Coverage ≥85% on `text/`, `parsing/`, `ingestion/`, `retrieval/`, `answering/`.
- Add a regression test for every bug found along the way.
- Probe tests for every §2 behaviour that production code depends on.

**Gate:** coverage targets met; every §2 claim has a test that would fail if the third party changed.

---

### P11 — Refactor and README `[S]` — **new, because presentation is graded**

`jobTask.md`: "Code readability, separation of concerns, error handling, and testing key components are important to us." This phase makes the work easy to review.

- **README** complete against all twelve items in §12, including known limitations stated honestly: no OCR and unusable Persian text layers, extractive answers cannot synthesize across distant passages, single-process deployment, small eval set, PyMuPDF licence note, and the §2.3 fixture gap.
- **`docs/DECISIONS.md` as the reviewer's entry point**, ordered so the model choice, thresholds, chunk size, and the §14 deviations are findable in one read, each with its evidence.
- **Test suite organized for review:** invariant tests grouped and runnable in one command (`pytest -m invariant`), so a reviewer can verify I1–I10 without reading the suite.
- **Dead code removed**; no file over roughly 400 lines; each module carries a one-line docstring stating its role and its layer.
- **Layer boundaries verified** by extending the AST purity test beyond `domain/` to confirm `api → services → domain ← adapters` holds.
- **Consistency pass:** error codes, config names, and docstrings agree with the OpenAPI schema and the README; every tunable lives in config; no magic numbers inline.
- Verify a clean clone: `uv sync && cp .env.example .env && make check && make run`.

**Gate:** Definition of Done (§13) fully checked.

---

## 10. Testing matrix

| Area | Required tests |
|---|---|
| Config and security | missing key, env and `.env` loading, `SecretStr` redaction, `fake` provider blocked outside test/demo, provider caps |
| Text | Persian variants, digits, ZWNJ, idempotence, identifiers, mixed scripts |
| Parsers | MD headings/fences/Setext, TXT encodings, PDF pages and blank pages, corrupt/scanned/encrypted PDF, size limits, magic bytes |
| Chunking | determinism, bounds, source coverage, slice identity, local-edit stability |
| Embeddings | order reassembly, non-permutation `index` fallback, three batching bounds, limiter with a fake clock, `Retry-After`, 429/5xx, auth, dimension, model-not-listed, cache |
| Storage | SQLite rollback, idempotent schema, `eligible_chunks`, hostile FTS input, **Chroma restart persistence**, cosine conversion, lock, dimension isolation |
| Diff | reused/added/removed, reorder, duplicates by multiset |
| Ingestion | add/unchanged/edit/delete/re-add, failure injection (embedder and Chroma), crash before publish, rebuild with 0 API calls, concurrency |
| Retrieval | dense-only, lexical-only, fusion, eligibility, doc and language filters, over-fetch |
| Gate and answer | answerable/unanswerable, threshold boundaries, verbatim-only, citation mapping, mutation test |
| API | every endpoint, status codes, validation, error shape, OpenAPI snapshot |
| Eval | runner, metric computation, reproducible report |

Invariants I1–I10 are release blockers. Coverage is a signal, not a substitute for them.

---

## 11. Coding standards

1. Type hints everywhere; `mypy src` clean under `strict`. Small pure functions, explicit dependency injection.
2. Network and storage I/O behind ports and adapters. No global mutable state for clients, repositories, or config.
3. No bare `except:`. Convert low-level exceptions to typed domain errors at architectural boundaries.
4. `async` for external network I/O; blocking SQLite and Chroma work in `asyncio.to_thread`.
5. Log identifiers, counts, timings, error classes — never document bodies or secrets.
6. Every tunable in config. No magic thresholds inline.
7. Every bug fix adds a regression test; every behavior change updates README, OpenAPI, and tests in the same commit.
8. Never silently fall back to another model, another version, or a fabricated answer.
9. Do not add an abstraction until a second implementation or a test seam requires it.

---

## 12. README requirements

1. Problem statement, constraints, assumption A1.
2. Architecture diagram (§4) and why there is **no** LLM generation: the service is embedding-only.
3. **Embedding model: BGE-M3, and why.** Latency, index size, multilingual and cross-lingual behaviour, and the measured quality spot-check, with the evidence's limits stated. Links `docs/eval_report.md`.
4. Storage design: SQLite as source of truth, FTS5 lexical, **local persistent Chroma** as a disposable derived index; data directory layout; backup by stopping and copying `data/`; `rebuild`.
5. Version publication and deletion consistency model: publish-then-cleanup, `eligible_chunks`.
6. Chunking and Persian normalization decisions.
7. Retrieval, evidence-gate, and answer-selection methodology; the meaning of `evidence_score`.
8. Evaluation dataset, measured metrics, calibration procedure and its limits.
9. API usage examples (`curl`) for upload, update, delete, query.
10. Setup (`uv`, env vars, **single-worker requirement**) and test instructions.
11. Known limitations: no OCR and unusable Persian text layers; extractive answers cannot synthesize across distant passages; single-process deployment; small eval set; PyMuPDF licence note.
12. Security notes: token externalized, redacted, never committed.

---

## 13. Definition of Done and final review checklist

- [ ] PDF, TXT, Markdown ingestion works; English fixtures, Persian per A1
- [ ] Add / update / delete via API; identical re-upload → zero embedding requests
- [ ] Local edits re-embed only changed chunks; reorder-only edits re-embed nothing
- [ ] Deleted and superseded content provably unretrievable (I1–I3), including with cleanup fault-injected
- [ ] Failed updates never expose a partial version (I3)
- [ ] Chroma local and persistent; restart persistence verified; single-instance lock enforced; `rebuild` restores dense search with zero API calls (I10)
- [ ] Hybrid retrieval measured against dense-only and lexical-only baselines
- [ ] Answers are exact source excerpts; every citation mechanically verified (I6, I7)
- [ ] Unanswerable questions → `insufficient_information`, no citations
- [ ] No GUI dependency; stable OpenAPI; structured errors; no secrets exposed (I8)
- [ ] Unit, integration, failure-injection, and concurrency tests green; `make check` clean
- [ ] `docs/eval_report.md`, `config/thresholds.json`, and `docs/DECISIONS.md` committed; README complete and honest about limits

Final review checklist, answered from tests:

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
- Can the embedding model change without mixing dimensions and collections, and are thresholds model-specific?
- Is the token absent from source control, logs, and error bodies?

---

## 14. Deviations from v2, with evidence

Every item here was a wrong assumption in v2, corrected by measurement in §2.

| v2 said | Reality | Change |
|---|---|---|
| "Reassemble vectors **by `index`**" | Gemini returns `index: [0,0,0,0]` | Accept `index` only when it is a permutation; else response order plus a warning (P4) |
| `MAX_CHARS_PER_REQUEST=160000` is the only size bound | A single 45 000-char input fails with `8192 token` context error | Add `MAX_CHARS_PER_ITEM=20000` (P4, §7) |
| 401/403 both mean auth failure | Unknown model → `403` **with the allowed-ID list** | `EmbeddingAuthError` covers it and surfaces available IDs, never the token (P4) |
| "Honor `Retry-After`, else backoff" | Header is absent on 429 | Backoff with jitter is the real path; honor the header if a provider ever sends it (P4) |
| Python ≥ 3.11 | Pinned `numpy==2.5.3` requires ≥3.12 | `requires-python = ">=3.12"`, ruff/mypy targets `py312` (P0, done) |
| Fixtures are the eval corpus | `tests/fixtures/docs/` was git-ignored; contains a 10 MB 312-page PDF | Author a small committed bilingual corpus; untrack the large PDF (§2.3) |
| Persian PDF parsing is solvable | `justforfun_book_a4.pdf` extracts in lossy presentation forms, unrecoverable | Document as a known limitation; use `ai-engineer.pdf`, which extracts correctly (§2.3) |
| Reviewer reads `DECISIONS.md` last | The job grades reasoning and readability explicitly | New P11 refactor pass, and `DECISIONS.md` becomes the reviewer's entry point |

---

## 15. Open items for the human

1. **Persian eval corpus** — confirmed bilingual by decision. Roughly 15% more effort than English-only; the eval is more honest for it.
2. **Fixtures** — `.gitignore` will stop excluding `tests/fixtures/docs/` wholesale; the 10 MB PDF is ignored by name, small fixtures are tracked.
3. **Live API during P9** — model comparison needs `RUN_LIVE=1` and the real token in `.env`. Everything else runs offline.
4. **Phases** — P0 is done and green. Next is P1; P6 remains the highest risk and gets the most attention.