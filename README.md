# Document Based Question Answering System (LLM-free extractive QA)

> **Status:** P0–P11 complete and green — 604 offline tests, 113 live against real BGE-M3.
> This file is the reviewer's entry point. Implementation detail lives in [`plan.md`](plan.md);
> the evidence log in [`docs/DECISIONS.md`](docs/DECISIONS.md) (D1–D77); the measurements in
> [`docs/eval_report.md`](docs/eval_report.md).

The service is **embedding-only**: the provider exposes `GET /v1/models` and
`POST /v1/embeddings` and **no chat endpoint**, so there is nothing to generate with. Answers
are **verbatim source slices with citations**, or an explicit `insufficient_information`.
Nothing paraphrases, summarizes, translates, or completes an answer.

---

## 1. The problem, and the constraints that shaped it

Upload documents, ask a question about them, get an answer that can be checked against the
document — with enough precision that "the model said so" is never the end of the reasoning.

Three constraints decided everything downstream:

- **The provider has no chat endpoint.** Only embeddings. So generation is not a degraded
  option, it is unavailable, and the system is *extractive* by necessity rather than by taste.
- **Outdated content must never be the basis of an answer.** Not "should not" — *never*, even
  when the vector store still holds the deleted vectors. That requirement alone decides the
  storage architecture (§5).
- **Evidence, not fluency.** An answer that cites a document which does not contain the answer
  is worse than no answer, because it spends the reviewer's trust. Every design choice below is
  a consequence of that sentence.

**Assumption A1 — Persian.** `jobTask.md` does not mention languages. The provider's locale is
Persian, so the system is bilingual by assumption: English first, Persian as a bounded second
stream, with a bilingual eval corpus. **Queries are English or Persian, and they match the
language of the documents.** Cross-lingual retrieval is *not* supported — see §3 and the limits
in §11. Some Persian PDFs in the wild have unrepairable text layers; those are detected and
reported rather than embedded as garbage. **No OCR** (out of scope).

---

## 2. Architecture

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

**Why there is no LLM generation.** Not a policy choice — the dependency has no chat endpoint.
The provider serves embeddings and nothing else, so a generating step would need a second
provider, a second dependency, and a new failure mode (a plausible sentence with no source).
Retrieval plus exact-substring selection gets to the same place with an answer you can verify
character by character.

**Dependency direction:** `api → services → domain/ports ← adapters`, enforced by an AST test
(`tests/unit/test_domain_purity.py`). `domain/` is pure stdlib; five layer rules and a cycle
detector keep the rest honest. Each rule exists because it was violated once (D76).

**Load-bearing decisions.**

- **SQLite is the single source of truth.** Publication state exists nowhere else.
- **Chroma is derived and rebuildable.** Vectors and three diagnostic fields only, never
  document text. `embedding_cache` retains every vector, so `rebuild` needs no API call (I10).
- **Publish is one transaction.** No second "is active" flag; eligibility is *derived* through
  a view, never remembered (I1).
- **Versions are monotonic per `doc_id`, across delete and re-add.** Vector ids are
  `"{doc_id}:v{version}:{ordinal}"`, so a stale vector can never collide with a fresh one.
- **Publish, then clean up.** Cleanup is best-effort, and its failure must not change what is
  retrievable.
- No message queue: the API shape does not need one.

---

## 3. The embedding model: BGE-M3, and why

`jobTask.md` offers four embedding models and asks us to choose the most appropriate one
and explain the choice. **BGE-M3 (`Bge-m3`) is the model.** This section is the only place in
the repository where the alternatives are named; `plan.md` carries only BGE-M3's measured
properties and the implementation decisions they force.

### Measured cost

Two-item embed request, same machine, same token (`plan.md` §2, D30):

| Model | Dimension | Latency | Relative | Index cost / 1 000 chunks (float32) |
|---|---|---|---|---|
| **BGE-M3** | **1024** | **0.64 s** | **1×** | **4.0 MB** |
| Embedding-3-Small | 1536 | 3.80 s | 5.9× slower | 6.0 MB |
| Embedding-3-Large | 3072 | 5.62 s | 8.8× slower | 12.0 MB |
| Gemini-embedding-001 | 3072 | 12.20 s | 19.1× slower | 12.0 MB |

BGE-M3 is simultaneously the smallest and the fastest. There is no dimension/latency
trade-off to make.

### Measured quality

The P9 eval dataset: **50 questions, 6 documents, 49 chunks, 11 285 characters, bilingual**.
Gold is an exact phrase that must appear in the retrieved chunk (chunking-independent,
`plan.md` §9.1), and a multi-section question is only scored as answered when **every** gold
phrase is in the window. Full report: [`docs/eval_report.md`](docs/eval_report.md).

| Strategy | R@1 | R@3 | R@5 | MRR@5 | false answers |
|---|---|---|---|---|---|
| dense only | 0.68 | 0.95 | 0.95 | 0.807 | 0 / 12 |
| lexical only | 0.58 | 0.76 | 0.87 | 0.687 | 0 / 12 |
| **hybrid (RRF, shipped)** | **0.76** | **0.95** | **0.95** | **0.846** | **0 / 12** |

Hybrid beats **both** single arms, which is what justifies the extra arm. Two earlier figures
are withdrawn: `R@1 = 1.000` measured on 75 chunks (D30) and the 12-question table above this
one (D59), replaced by these.

**The evidence gate, calibrated.** Thresholds are grid-searched on the eval dataset's **dev
split** and reported on the held-out split (`plan.md` §9.3: maximise answered-on-answerable
subject to a false-answer rate within 5%, preferring 0%):

| split | answered on answerable | **false answers** | gold quoted in the cited text |
|---|---|---|---|
| dev (23 answerable / 7 unanswerable) | 0.913 | **0** | 0.913 |
| held out (15 / 5) | 0.933 | **0** | 0.875 |

Shipped in `config/thresholds.json` as `min_dense=0.54`, `min_coverage=0.40`,
`min_coverage_high=0.75`, `min_sentence_overlap=0.15`, with `calibrated: true`, the split, the
class balance and the dataset size recorded alongside the numbers, and refused outright if the
model ever changes (I9). **50 questions over six documents is a coarse instrument** — one
question is worth 0.05 recall — so these are operating points, not constants.

**A 0% false-answer rate is not the same as being right.** `gold quoted` — the share of gold
phrases that appear in a chunk the answer actually cites — is the metric that separates them,
and it exists because §9.2's list could not: an answered question citing a document that does
not contain the answer passes the refusal rate, the R@k, *and* the citation-validity check. It
is the **acceptance metric for requirement 4**, because a citation asserts its source supports
the answer — substring validity ("the quote is faithful") is a different and weaker claim.

Every failing case is named with its cause in [`docs/eval_report.md`](docs/eval_report.md):
three cite the wrong document, one is a `top_k` truncation (the gold chunk sits at window
position 10, below `top_k=5`), and one is a sentence the overlap bar correctly refuses — it
scores 0.067 against the 0.15 bar while the sentence beside it, which talks *about* the port
without containing the number, scores 0.153 and gets quoted instead. **No target number is
set**: the target is to remove the causes (D66, D68).

### Why the choice holds

1. **Multilingual, which is a requirement here.** The corpus and the queries are bilingual, and
   one multilingual model removes a whole class of bugs that invariant I9 would otherwise have
   to police — misrouted queries, language filters, per-language collections. `detect_language`
   is a stored *label*, not a routing decision; one collection serves both languages.

   **What that does and does not buy, measured.** BGE-M3 genuinely aligns an English question
   to a Persian source with no translation step. What the system *ships* is narrower: the same
   English question is **refused** against a 7-chunk Persian index and **answered** against
   49- and 1225-chunk ones, with `max_dense` flat at 0.513–0.547 across all three. Absolute
   cosine does not drift with corpus size; what changes is token coverage, which counts
   *shared* tokens and is therefore structurally zero when nothing else in the corpus happens
   to share a word. Cross-lingual retrieval here works when an unrelated document lends it
   lexical corroboration, and not otherwise — **luck rather than capability** (D61).

   The uncorroborated dense branch that could have carried it was **removed** (D71): no
   available second signal separates a real cross-lingual hit from a near-topic one on this
   dataset, and keeping it would have cost the 0% false-answer rate that requirement 5 asks
   for. Those are the same choice, not alternatives (§11, D61/D69/D71). A hit sharing no token
   with the question is now refused. Making cross-lingual retrieval work is a **dataset and
   signal** problem, not a threshold-tuning one.
2. **Cheapest at the quality we measured.** Smallest dimension, lowest latency, and no measured
   retrieval gain available from the larger models on this corpus. Their extra cost buys
   nothing here.
3. **Fits the quota with room to spare.** At 120 req/min, ingest cost is dominated by request
   count, not characters. A full re-ingest of the 393k-char Persian book measured **31
   req/min** against our 100/min limiter, so re-ingest and `rebuild` never become rate-limit
   problems.
4. **A single 1024-d space keeps the index small**: 4 MB per 1 000 chunks against 12 MB, which
   matters for a local on-disk Chroma index.

### One implementation detail worth knowing

BGE-M3's `index` field in the response is a genuine permutation, but the provider's
OpenAI-compatible contract does not guarantee that. The client validates it and falls back to
response order when it is not a permutation (D21) — cheap insurance against a response shape
that would otherwise silently mislabel every vector.

---

## 4. Storage design

**SQLite is the source of truth. Chroma is a derived, disposable index.** That split is the
whole design, and it is what makes requirement 2 satisfiable.

| store | holds | if it is deleted |
|---|---|---|
| `data/qasystem.db` (SQLite) | `documents`, `document_versions`, `chunks`, `chunks_fts` (FTS5), `embedding_cache`, `ingest_log`, and the `eligible_chunks` view | everything is lost |
| `data/chroma/` (Chroma, `PersistentClient`) | one vector per chunk, plus `doc_id`, `doc_version`, `ordinal` | **nothing is lost.** `python -m qasystem.cli rebuild` restores dense search from `embedding_cache` with **zero** API calls (I10) |

- **No document text in Chroma.** Metadata is three fields; the text lives in SQLite only. A
  vector store that held the prose would be a second copy to keep in sync.
- **`embedding_cache` retains every vector**, keyed by `(sha256(exact embedded string),
  model_id)`. That single row set is what makes both I5 (an unchanged chunk costs zero
  embedding requests) and I10 work. Keying on the model is I9: two models must never read each
  other's vectors, even at the same dimension.
- **Collection name carries identity**: `chunks__{slug(model_id)}__d{dimension}`, and
  `model_id`/`dimension` are also stored in collection metadata and **asserted on open** — the
  name can collide after truncation, so the metadata is the real guard (D27).
- **Cosine space**, and `similarity = 1 − distance`, proven by a known-vector test rather than
  assumed.
- **SQLite pragmas** `journal_mode=WAL`, `foreign_keys=ON`, `busy_timeout`, with FTS5 presence
  checked at startup rather than assumed.
- **`ingest_log`** carries counts and timings only — never bodies, text, or secrets.

**Backup: stop the app and copy `data/`.** Never mutate Chroma's files directly. `data/` is
git-ignored. A single process owns it: `uvicorn --workers 1`, taking `FileLock(timeout=0)` at
startup, so a second instance fails fast instead of interleaving writes.

---

## 5. Version publication and deletion consistency

This is the part that requirement 2 is really about, and the part where "best effort" is
dangerous. A deleted document must be unretrievable **even if the vector cleanup never ran**.

**Every retrieval path reads chunks through one view.** Nothing else decides eligibility:

```sql
-- THE choke point. Every retrieval path reads chunks ONLY through this view (I1, I2).
CREATE VIEW eligible_chunks AS
  SELECT c.* FROM chunks c
  JOIN documents d ON d.doc_id = c.doc_id
                  AND d.status = 'active'
                  AND d.current_version = c.doc_version;
```

**Publish is one SQLite transaction.** There is no second "is active" flag, so eligibility is
derived and never remembered. A failure *before* publish leaves the previous version fully
queryable and unchanged (I3).

**Delete is one transaction** setting `status='deleted'` and `current_version=NULL`. From that
commit the document is gone from both retrieval arms — lexically and densely — because the view
stops returning its rows. Chroma vectors are purged best-effort *afterwards*; that purge's
failure cannot make a deleted document retrievable, and there is a test that injects exactly
that failure (I2). The `documents` tombstone is kept so `last_version` stays monotonic across
delete and re-add.

**Ingest flow:** validate → parse → chunk → hash → *unchanged? return early with zero embedding
requests* → diff against the published chunks by `chunk_hash` → embed only the missing inputs
→ stage (SQLite txn) → stage (Chroma) → **verify** the vector set → **publish** (one SQLite
txn) → **clean up** (best-effort, separately asserted).

Two details worth naming:

- **Diff is a multiset diff** (hash → queue), so duplicate chunks keep their multiplicity and a
  pure reordering costs **zero** embedding calls.
- **Embedding happens before any version exists and outside the write lock**, so an embedder
  failure costs nothing and leaves no half-published state.

---

## 6. Chunking and Persian normalisation

- **Contiguous slices only.** Every chunk is `source_text[char_start:char_end]`, which is what
  makes I6 mechanically checkable instead of aspirational: a test asserts the identity for
  every segment of every answer.
- **Deterministic and heading-aware.** No randomness, no reordering — the same document always
  produces the same chunk ids and hashes.
- **`CHARS_PER_TOKEN = 1.5` is the *densest* measured case** (tables: 1.45, prose: 4.14), not
  the average (D17). Sizing on prose would overflow the model's item ceiling on a table.
- **Sections are chunked independently and never merged** (D18). Merging across a heading
  boundary produces chunks whose text is not contiguous in the source.
- **A sentence-free page is cut mechanically** (D19). A second cap pass exists for content with
  no sentence terminators at all, so a code block or a table cannot produce an oversized chunk.
- **Over-sized chunks are refused locally, before any request** (D22). `MAX_CHARS_PER_ITEM =
  20000` against a measured 40 949-char ceiling is a 2.05× margin; truncating instead would
  embed a vector for text nobody could cite (I6).

**Persian normalisation.** NFKC, plus Persian letter and digit folding, applied **in the PDF
parser** rather than to stored text (D14) — we construct that text ourselves, so normalising it
changes the bytes the citations point into. Stored text is never rewritten.

**ZWNJ is kept, and all three of its forms are indexed** (D8): the compound (`می‌رود`), the
fused (`میرود`), and the parts (`می`, `رود`). One spelling is not enough — a document written
one way and a query written another must still match, and a D20 regression deleted ZWNJ
entirely until a test caught it. Hidden characters are Unicode category `Cf`; one category test
beats a hand-listed table, but ZWNJ and ZWJ must stay in the keep-set.

**No RTL reversal.** Mangling a citation is worse than saying the text is mangled.

---

## 7. Retrieval, the gate, and answer selection

**Retrieval — hybrid, fused over the candidate union.**

1. Normalise and tokenise the question with the *same* functions used for documents. Empty or
   whitespace → 422.
2. **Dense:** embed the query → `query(n = CANDIDATES_N × OVERFETCH)` against local Chroma.
3. **Lexical:** FTS5 `bm25` over `eligible_chunks`, top `CANDIDATES_N`.
4. **Eligibility:** join dense ids to `eligible_chunks` in **one statement**, applying
   `doc_ids` and `language` and fetching chunk text in that same statement. Anything not
   returned is dropped (I1, I2).
5. **Fusion:** weighted RRF over the **union** of the two lists —
   `score = w_d/(k+rank_d) + w_l/(rank_l)`, with `DENSE_WEIGHT=0.9`, `LEXICAL_WEIGHT=0.1`,
   `RRF_K=60`, `TOP_K=5`.

Fusing *full-corpus* dense ranks instead would let one rank-1 lexical hit outrank a correct
dense hit — it measures a system nobody will build and it looks exactly like a finding (D30).
Fusing the candidate window is what makes single-arm baselines possible at all: a weight of
**0 disables** an arm, so `dense_weight=0` is a real dense-only system rather than a silent mix
(D39). Ranks are 1-based everywhere, so the best hit never divides by `k` (D38).

**The evidence gate runs *before* any answer text is chosen.** Signals: `max_dense_similarity`,
normalised lexical score, and `token_coverage` (the share of non-stopword query tokens present
in the best chunk). The rule, keyed by `model_id` so thresholds can never be reused across
models:

```text
passed = (max_dense ≥ min_dense AND coverage ≥ min_coverage)
         OR (coverage ≥ min_coverage_high)
```

Coverage is the **maximum over candidates**, not the top candidate's (D41) — so a fusion bug
cannot present itself as a gate refusal. Thresholds for another model are a hard error; a
*missing* file falls back to uncalibrated defaults (D42, D51).

**Below threshold** → `status="insufficient_information"`, `citations=[]`, a fixed localised
system message (English and Persian, from a small table), and a machine-readable `reason` ∈
`{empty_knowledge_base, no_relevant_content, below_threshold}`. The message is not evidence and
is never cited, and "closest" chunks are **never** presented as citations.

**Answer selection** (`MAX_ANSWER_SENTENCES=5`) scores sentences by IDF-weighted query-token
overlap plus a small retrieval-score term; a sentence must clear `min_sentence_overlap=0.15`
to be eligible, so a relevant chunk is never padded with unrelated sentences. Selections are
ordered by source position and merged into contiguous slices. Output is **segments** —
`{text, citation_id, chunk_char_start, chunk_char_end}` with `text == chunk.text[start:end]` —
and the `answer` string is a *rendering* of them (`text [n]`), where `[n]` markers are system
metadata placed outside the slices.

**`evidence_score` is a deterministic evidence score, not a probability.** It is named to avoid
`confidence` on purpose: it is not calibrated, it does not sum to 1, and it is not comparable
across questions. Do not threshold on it as if it were one.

A citation carries `{id, doc_id, document, doc_version, section, page, lines, chunk_id,
char_span, excerpt, score}`, so every claim points at a version, a section, a page or line range,
a character span in the stored source, and the exact excerpt.

---

## 8. Evaluation, calibration, and what they can tell you

**The dataset** (`tests/eval/`): 6 documents of 1–3 KB covering md, txt and pdf in both
languages; 50 questions — 15 answerable English, 10 answerable Persian, 8 exact-term, 5
multi-section, 8 unanswerable near-topic, 4 unanswerable off-topic. Gold is an exact phrase
from the source, never a chunk id, so the dataset survives chunking changes. Split dev/held-out
60/40 **before** looking at results.

**Metrics** for dense-only, lexical-only and hybrid: R@1/3/5, MRR@5, unanswerable false-answer
rate, answerable refusal rate, `gold quoted`, citation substring validity (must be 100%),
stale-content leakage (must be 0), P50/P95 latency, index size, ingest request count.

**Calibration** maximises answered-on-answerable subject to a false-answer rate within 5%,
preferring 0% — because the task is evidence-first. The objective, split, class balance,
dataset size and evidence limit are written *into* `config/thresholds.json` next to the
numbers, so the file cannot be read as a constant.

Three harness lessons that changed published numbers, recorded because each would have
published a wrong figure:

- `embed_requests` counted `embed()` calls rather than HTTP requests — a **32× error** in a
  number the plan reports (D46).
- A latency measurement reported 11 ms because the query cache was warm; latency is now
  measured on a fresh index every run (D63).
- A grid search reported "no feasible point" while 26 040 feasible points existed, because of
  its own bounds (D57).

**What the eval cannot tell you.** 50 questions over six documents is a small instrument: one
question is worth 0.05 recall and a single flipped answer moves a rate by a quarter. The
numbers are operating points on a specific corpus, not constants, and no quality claim is made
about any model that was not benchmarked under the final configuration. No pricing claim is
made at all — none was available.

---

## 9. API

Interactive documentation is served at **`http://127.0.0.1:8000/docs`** (Swagger UI) and
`/redoc`; the schema is at `/openapi.json`. The examples below are the same calls in `curl`,
for a reviewer who does not have the page open.

**Upload** — `201` created · `200` unchanged · `409` exists with different content:

```bash
curl -sS -X POST http://127.0.0.1:8000/documents \
  -F file=@handbook.md \
  -F doc_id=handbook
```

**Update** (replaces an *active* document; `404` for unknown or deleted):

```bash
curl -sS -X PUT http://127.0.0.1:8000/documents/handbook -F file=@handbook-v2.md
```

**Delete** — `204`; `404` if unknown or already deleted:

```bash
curl -sS -X DELETE http://127.0.0.1:8000/documents/handbook
```

**Query** — `answered` or `insufficient_information`:

```bash
curl -sS -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the service return when the token is invalid?"}'
```

```json
{"status": "answered",
 "answer": "The service returns ERR-404 when the token is invalid. [1]",
 "segments": [{"text": "The service returns ERR-404 when the token is invalid.",
               "citation_id": 1, "chunk_char_start": 0, "chunk_char_end": 54}],
 "evidence_score": 0.824,
 "citations": [{"id": 1, "doc_id": "handbook", "document": "handbook.md", "doc_version": 1,
                "section": "Handbook > Usage", "page": null, "lines": [9, 9],
                "chunk_id": "handbook:v1:1", "char_span": [99, 153],
                "excerpt": "The service returns ERR-404 when the token is invalid.",
                "score": 0.0164}],
 "reason": null, "debug": null}
```

That is a real response, trimmed only in whitespace. Two things in it are easy to misread, so
they are worth naming: a citation's `score` is the **fused RRF rank score** and is therefore
small (0.0164) — it is not a confidence and not comparable across questions; and
`evidence_score` (0.824) is a separate, deterministic figure. Neither is a probability.

`answer` is a *rendering* of `segments`: the `[1]` markers are system metadata placed outside
the slices, and each segment's `text` is exactly `chunk.text[chunk_char_start:chunk_char_end]`.
The response is `insufficient_information` with `citations: []` and a `reason` when the gate
declines.

A Persian question, filtered to specific documents, with gate diagnostics:

```bash
curl -sS -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "چه چیزی هنگام نامعتبر بودن توکن برمی‌گرداند؟",
       "doc_ids": ["handbook-fa"], "language": "fa", "debug": true}'
```

`debug: true` adds a `debug` object with the gate's reasoning — `passed`, `reason`,
`max_dense`, `lexical`, `token_coverage`, `overlap`, `candidate_count`, `thresholds_version`,
`calibrated`, `model_id` — and the candidate list. **`debug` never widens what is returned**:
it explains a decision, so it cannot leak a chunk the non-debug response would have hidden,
and it cannot turn a refusal into an answer (D53).

Other endpoints: `GET /documents` (paginated), `GET /documents/{doc_id}` (metadata, current
version, section outline), `GET /health` (liveness, always 200 — deliberately silent about
dependencies, so a database outage does not become a restart loop), `GET /ready` (SQLite, FTS5,
Chroma, the data-dir lock, the model and dimension, and whether thresholds are calibrated;
503 otherwise).

**One error shape, always:**

```json
{"error": {"code": "DOCUMENT_NOT_FOUND", "message": "no active document with id 'handbook'", "request_id": "3f9c…"}}
```

Unknown exceptions become `500 INTERNAL_ERROR` with no stack trace and no detail. Every
response carries an `X-Request-ID`; the access log records the path, the status, the duration
and the error class — never a body, never a header value, never a query string, because a URL
can carry a question and a question can carry a secret (D53).

**CLI:** `python -m qasystem.cli reconcile | rebuild | check-storage | calibrate | eval`.

---

## 10. Setup and tests

```bash
uv sync
cp .env.example .env      # then set EMBEDDING_API_KEY and EMBEDDING_MODEL=Bge-m3
make check                # ruff check + ruff format --check + mypy src + pytest, fully offline
make run                  # uvicorn --factory --workers 1
```

**A single worker is required.** One process owns `data/` and takes `FileLock(timeout=0)` at
startup; a second instance on the same `data/` refuses to start rather than interleaving writes
(I7/L2, verified with real subprocesses).

**`make run` needs the real provider.** With `EMBEDDING_PROVIDER=fake` the app refuses to start,
and that is the invariant working rather than a fault: the shipped `config/thresholds.json` is
calibrated for `Bge-m3`, and I9 will not reuse one model's measurements for another — the fake
embedder reports `model_id="fake-embedder"`. To try the system without a token, either point
`THRESHOLDS_PATH` at a thresholds file calibrated for `fake-embedder`, or drive the HTTP surface
from the test suite, which uses the fake provider throughout.

| command | what it does |
|---|---|
| `make check` | the offline gate: lint, format, `mypy --strict`, 604 tests |
| `make live` | 113 tests against **real** BGE-M3. Not optional — see below |
| `make check-live` | both, which is the full definition of done |
| `make invariants` | the 23 tests that witness I1–I10, in 1.5 s |
| `make eval` / `make calibrate` | the metrics / the grid search, against the real provider |

`make live` is part of the definition of done because **the fake embedder cannot catch the
defects that matter.** It is a bag of hashed tokens: it cannot rank, cannot align languages,
and its similarity scale is nothing like the real model's. P0–P7 passed 480 offline tests while
the gate was unable to authorise the system's own cross-lingual capability (D47), a Latin word
glued to Persian script was unmatchable (D45), `embed_requests` was 32× wrong (D46), and every
endpoint 500'd on a thread-bound SQLite connection (D49). Offline green proves the plumbing;
only live proves the product.

Single test, single file, filtered:

```bash
uv run pytest tests/unit/test_gate.py -q
uv run pytest tests/integration/test_retrieval.py::test_3b_the_rendered_answer_is_exactly_the_slices_plus_their_markers -q
uv run pytest -k "invariant or persian" -q
```

Network is blocked in tests by `socket.connect` monkeypatching — a test that opens a socket
**fails** rather than skipping, so nothing quietly depends on the network. Live calls need
`RUN_LIVE=1`, and `make live` runs `tests/integration` only; the 7 live HTTP tests in
`tests/api/test_api_live.py` need `RUN_LIVE=1 uv run pytest` over the whole suite.

**There is no CI.** The Makefile is the gate, and it only runs when someone runs it.

---

## 11. Known limitations

Stated rather than hidden, because each one bounds a claim above.

- **No OCR.** A PDF whose text layer is empty or unusable is rejected with `NO_TEXT_LAYER`, not
  read. Some Persian PDFs are unusable this way: the committed 749 KB fixture maps the space
  glyph to nothing on 274 of 6 563 content lines, so those arrive as one run of letters. NFKC
  recovers the letters and their order; **it cannot invent a space the PDF never stored.**
  Repairing that needs Persian word segmentation, which is out of scope. Retrieval still works
  on the characters that are present.
- **Cross-lingual retrieval is not supported.** §3 and §11's decision D71: a hit sharing no
  token with the question is refused, because no second signal could make accepting it safe
  without giving up the 0% false-answer rate. An English question against a Persian document
  works only when some other document lends lexical corroboration. 2 of the 4 cross-lingual
  eval questions are answered; the other 2 are refused with the chunk retrieved and
  semantically close.
- **Extractive answers cannot synthesise across distant passages.** The answerer selects
  sentences that individually contain query terms. A question whose answer requires combining
  three sections will find the section with the best overlap, not the three sections together.
  A multi-section question is scored as answered only when *every* gold phrase is in the window.
- **Word overlap cannot bridge a question asked in words and an answer given as a literal.**
  `"what port"` never reaches `0.0.0.0:8443`, and `"which command"` never reaches a Markdown
  code fence (D68). Raising `TOP_K` was measured and rejected: 0.897 → 0.923 for +19% answer
  length.
- **The eval set is small**, 50 questions over 6 documents. Coarse operating points (§8).
- **The fixture corpus is skewed**: one 749 KB Persian book is 84% of the index, so fixture
  numbers are not representative. This is why `make eval` builds its own index in `data/eval/`
  rather than ingesting the fixtures alongside it — the skew would have measured itself.
- **Chunk size is unresolved.** The eval corpus's longest section (635 chars) is below the
  smallest hard cap tried, so every chunk-size row is identical *by construction* and the
  experiment cannot discriminate (D64). No change was adopted because nothing was measured.
  The cost half stands — 105 ms per 1 000 chars at 500-char items, 11 ms at 8 000.
- **Query latency is the provider's, not ours**: P50 354 ms cold on a fresh index, of which
  our own retrieval, fusion, gating and assembly are ~3% (P50 10 ms warm, D54/D63). Any P95
  quoted without that qualification is measuring someone else's server.
- **Single-process deployment.** One worker, one data directory, no queue, no horizontal
  scaling. `reconcile` at startup repairs index drift; it is not a clustering story.
- **Answer selection is not a summariser**, so a long answer may repeat itself across adjacent
  sentences, and `MAX_ANSWER_SENTENCES` trades recall for length.
- **Persian morphology is unhandled** beyond ZWNJ and NFKC: no stemmer, no Ezafe-aware
  matching. Token coverage loses 0.29 to Ezafe suffixes (D43), which the calibration absorbed
  by setting `min_coverage` to 0.40 rather than 0.25 — measured, not assumed.
- **PyMuPDF** (`pymupdf==1.28.2`, AGPL) is the PDF parser. It is the only dependency here with a
  copyleft licence, which matters if this is ever distributed rather than run internally.

---

## 12. Security

- **The token is externalised and never committed.** `EMBEDDING_API_KEY` is read from the
  environment only, into a pydantic `SecretStr`. `.env` is git-ignored; `.env.example` holds
  placeholders. It never appears in `repr`, in serialisation, in a log line, in an error body,
  or in a test snapshot.
- **Provider errors are scrubbed before they leave the client** (D23). The provider's own
  message is where the model list and the rate-limit caps live, so it is genuinely useful — and
  a provider that echoes the `Authorization` header must not turn our error body into a leak.
  The same scrub runs on the logging handler and on domain-error messages returned to a client.
- **The access log records identifiers, counts, timings and error classes only** — never a
  body, never a header value, **never the query string**, because a URL can carry a question
  and a question can carry a secret (D53).
- **Unknown exceptions are flattened** to `500 INTERNAL_ERROR` with no stack trace, no detail,
  and no provider text. Clients see a `request_id`; the detail stays in the log.
- **No document text in the vector store**, and no text in `ingest_log`, so the persisted index
  carries no content beyond what SQLite already holds under the same access controls.
- **The suite cannot leak by accident.** `tests/unit/test_repo_consistency.py` fails if any
  fixture the suite reads is untracked (D31), and `test_no_endpoint_echoes_a_secret` plus the
  redaction tests cover the paths above. Storage tests use `tmp_path` only — never the real
  `data/`.
- **No authentication on the API.** It is a single-process internal service and auth was out of
  scope. Do not expose it to an untrusted network without putting authentication in front of it.

---

## Documentation map

| Document | Contains |
|---|---|
| [`plan.md`](plan.md) | Requirements, BGE-M3's measured properties, architecture, invariants I1–I10, phases, engineering rules, Definition of Done |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | D1–D77, each with its measurement or mutation evidence, plus a reviewer's index at the top |
| [`docs/eval_report.md`](docs/eval_report.md) | P9: 50-question metrics, the weight and chunk-size experiments, the calibration, and the cases that cite the wrong source |
| [`AGENTS.md`](AGENTS.md) | Commands, gates, configuration gotchas and testing quirks for anyone working *in* the repository |
| [`refactor-plan.md`](refactor-plan.md) | The P11 audit and the order its findings were fixed in |