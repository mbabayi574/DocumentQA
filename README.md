# Document Based Question Answering System ( LLM-free extractive QA system )

> **Status:** P0–P5 complete and green (313 tests). P6–P11 pending.
> This file is the reviewer's entry point. Implementation detail lives in [`plan.md`](plan.md)
> and the evidence log in [`docs/DECISIONS.md`](docs/DECISIONS.md).

The service is **embedding-only**: the provider exposes `GET /v1/models` and
`POST /v1/embeddings` and no chat endpoint, so there is nothing to generate with. Answers
are **verbatim source slices with citations**, or an explicit
`insufficient_information`. Nothing paraphrases, summarizes, translates, or completes.

---

## The embedding model: BGE-M3, and why

`jobTask.md` offers four embedding models and asks us to choose the most appropriate one
and explain the choice. **BGE-M3 (`Bge-m3`) is the model.** This section is the only place
in the repository where the alternatives are discussed; `plan.md` contains only BGE-M3's
measured properties and the implementation decisions they drive.

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

Hybrid beats **both** single arms, which is what justifies the extra arm. Two earlier figures are
withdrawn: `R@1 = 1.000` measured on 75 chunks (D30) and the 12-question table above this one
(D59), replaced by these.

**The evidence gate, calibrated.** Thresholds are grid-searched on the eval dataset's **dev
split** and reported on the held-out split (`plan.md` §9.3, objective: maximise answered-on-
answerable subject to a false-answer rate within 5%, preferring 0%):

| split | answered on answerable | **false answers** | gold quoted in the cited text |
|---|---|---|---|
| dev (23 answerable / 7 unanswerable) | 0.91 | **0** | 0.91 |
| held out (15 / 5) | 0.93 | **0** | 0.88 |

Shipped in `config/thresholds.json` with `calibrated: true`, the split, the class balance and
the dataset size alongside the numbers, and refused outright if the model ever changes
(`plan.md` §5.1 I9). **50 questions over six documents is a coarse instrument** — one question
is worth 0.05 recall — so these are operating points, not constants.

**A 0% false-answer rate is not the same as being right.** `gold quoted` is the metric that
distinguishes them, and it exists because §9.2's list did not: an answered question that cites
a document which does not contain the answer passes the refusal rate, the R@k, and the citation
*validity* check. On the held-out split 12% of the facts the system claims to have answered are
not in the text it quotes.

**`gold quoted` is the acceptance metric for requirement 4** — a citation asserts its source
supports the answer, so substring validity ("the quote is faithful") is not the same claim as "the
source contains the answer". Every failing case is named with its cause in
[`docs/eval_report.md`](docs/eval_report.md): three cite the wrong document, one is a `top_k`
truncation (the gold chunk sits at window position 10, below `top_k=5`), and one is a sentence
the overlap bar correctly refuses — it scores 0.067 against a 0.15 bar while the sentence beside
it, which talks *about* the port without containing the number, scores 0.153 and is quoted
instead. No target number is set: the target is to remove the causes.

### Why the choice holds

1. **Multilingual, which is a requirement here.** The corpus and the queries are bilingual.
   Cross-lingual English-question → Persian-source retrieval works with no translation
   step, no language detector, and no per-language index. A single multilingual model
   removes an entire class of bugs — misrouted queries, language filters, per-language
   collections — that invariant I9 would otherwise have to police.

   *Live-verified — and the P9 measurement showed the property does not hold as stated, so it
   is worth being precise about what works.* The same English question was asked against three
   indexes: it is **refused** on a 7-chunk Persian index and **answered** on 49- and
   1225-chunk ones, with `max_dense` flat at 0.513–0.547 across all three. Absolute cosine does
   not drift with corpus size; what changes is token coverage, which counts *shared* tokens and
   is therefore structurally zero when nothing else in the corpus happens to share a word. So
   cross-lingual retrieval on this system works when an unrelated document lends it lexical
   corroboration, and not otherwise — luck rather than capability. The arithmetic is in
   `plan.md` §12 question 3: refusing the hardest unanswerable case needs `min_dense_alone >
   0.609` and the real cross-lingual hits measure 0.513–0.547, so the branch cannot be both
   reachable and safe. That question is open, with the measurements in front of it (D61).
   What P9 did fix: the **1-in-9 false-answer rate does not reproduce** on a balanced corpus
   where every question has a known answer — it is 0 of 12 unanswerable, on both splits.
2. **Cheapest at the quality we measured.** Smallest dimension, lowest latency, and no
   measured retrieval gain available from the larger models on this corpus. Their extra
   cost buys nothing here.
3. **Fits the quota with room to spare.** At 120 req/min, ingest cost is dominated by
   request count, not characters. A full re-ingest of the 393k-char Persian book measured
   **31 req/min** against our 100/min limiter, so re-ingest and `rebuild` never become
   rate-limit problems.
4. **A single 1024-d space keeps the index small**: 4 MB per 1 000 chunks against 12 MB,
   which matters for a local on-disk Chroma index.

### Limits of this evidence, stated plainly

- **50 questions over six documents is still a small eval.** One question is worth 0.05
  recall and a single flipped answer moves a rate by a quarter. The thresholds are coarse
  operating points and the file says so.
- **Cross-lingual retrieval is weaker than the headline claim**, as measured above (D61).
  2 of the 4 cross-lingual questions in the eval dataset are answered; the other 2 are refused
  by the cosine bar with the chunk retrieved and semantically close.
- **Chunk size was not resolved.** The eval corpus has 49 sections in 49 chunks with a longest
  section of 635 characters, below the smallest hard cap tried, so every chunk-size row is
  identical *by construction* and the experiment cannot discriminate (D64). The cost half
  stands: 105 ms per 1 000 chars at 500-char items, 36 ms at 2 000, 11 ms at 8 000 — 2.9× and
  9.5× cheaper. No change was adopted, because nothing was measured.
- **Query latency is the provider's, not ours.** Measured cold on a fresh index: P50 354 ms,
  P95 448 ms. The same queries with the embedding cached take P50 10 ms, so retrieval, fusion,
  gating and answer assembly together are ~3% of a query (D54, D63). Any P95 quoted without
  that qualification is measuring someone else's server.
- **`min_lexical` was a threshold on a saturated signal and has been removed** (D67, D58):
  `lexical_score` is bm25 divided by the best bm25 of the same query, so the top hit was exactly
  1.0 for every query FTS matched — 1.00 for 49 of the 50 eval questions, answerable and
  unanswerable alike. The gate's exact-terms branch is now decided by `min_coverage_high` alone,
  which is provably the same rule, and the calibration's operating point is unchanged.
- **Two remaining answers quote a document that does not contain the answer** (`gold quoted`
  0.90 on all 50 questions). Both are the cross-lingual cases above, and both trace to the same
  mechanism. The other two of the four cases are a single defect — word overlap cannot connect a
  question asked in words to an answer given as a literal, which is why `"what port"` never reaches
  `0.0.0.0:8443` and `"which command"` never reaches a Markdown code fence (D68). Raising `TOP_K`
  was measured and rejected: 0.897 → 0.923 for +19% answer length.
- **No quality claim is made about any model that was not benchmarked under the final
  configuration**, and **no pricing claim is made at all** — none was available.
- Document size skews dense retrieval: on a corpus capped at 40 chunks per document R@1 was
  0.75, against 0.67 on the full corpus where one book is 84% of the index. This is why the
  eval runs against **its own index** in `data/eval/` rather than alongside the fixtures — the
  fixture skew would have measured itself.

### One implementation detail worth knowing

BGE-M3's `index` field in the response is a genuine permutation, but the provider's
OpenAI-compatible contract does not guarantee that. The client validates it and falls
back to response order when it is not a permutation (D21) — a cheap guard against a
response shape that would otherwise silently mislabel every vector.

---

## Documentation map

| Document | Contains |
|---|---|
| [`plan.md`](plan.md) | Requirements, BGE-M3 capabilities, architecture, invariants, phases P6–P11, coding standards, Definition of Done |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | D1–D72: every decision with its measurement or mutation evidence |
| [`docs/eval_report.md`](docs/eval_report.md) | P9: 50-question metrics, the weight and chunk-size experiments, the calibration, and the five answers that cite the wrong source |

## Quick start

```bash
uv sync
cp .env.example .env      # add EMBEDDING_API_KEY, EMBEDDING_MODEL=Bge-m3
make check                # ruff + mypy + pytest, fully offline
make run                  # uvicorn --workers 1  (a single worker is required)
                           # then open http://127.0.0.1:8000/docs for Swagger UI

# P9, against the real provider (needs EMBEDDING_API_KEY; writes its own data/eval index)
make eval                 # metrics for hybrid vs dense-only vs lexical-only  -> docs/eval_report.md
make calibrate            # the same, plus grid-searched thresholds -> config/thresholds.json
make live                 # the test suite against the real provider
```

See [`plan.md`](plan.md) §7 for every tunable and §13 for the release checklist.
