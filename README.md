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

Committed fixture corpus, 1 225 chunks from 5 documents, 12 answerable questions, gold =
an exact phrase that must appear in the retrieved chunk (chunking-independent, `plan.md` §9.1):

| Strategy | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| dense only | 0.67 | 0.83 | 0.92 | 0.757 |
| lexical only | 0.42 | 0.67 | 0.75 | 0.531 |
| hybrid (RRF) | 0.67 | 0.92 | 0.92 | 0.764 |

An earlier internal figure of `R@1 = 1.000` was measured on 75 chunks from three small
documents and **did not reproduce**; it is withdrawn (D30). The numbers above are the
honest ones, and they are the reason the system is hybrid and gated rather than
dense-and-answer.

### Why the choice holds

1. **Multilingual, which is a requirement here.** The corpus and the queries are bilingual.
   Cross-lingual English-question → Persian-source retrieval worked with no translation
   step, no language detector, and no per-language index. A single multilingual model
   removes an entire class of bugs — misrouted queries, language filters, per-language
   collections — that invariant I9 would otherwise have to police.
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

- One corpus, 5 documents, 12 answerable and 2 unanswerable questions. **Far too small to
  certify a false-answer rate**; the P9 dataset (≥50 questions, 60/40 split) exists for
  that.
- One question is worth 0.083 R@1 at this sample size, so the hybrid-weight difference
  below is *not* yet significant.
- No ablation over chunk size or fusion weights. The largest measured efficiency finding
  is that cost per character falls from 105 ms per 1 000 chars at 500-char items to 36 ms
  at 2 000 and 11 ms at 8 000 — 2.9× and 9.5× cheaper — but raising chunk size trades
  throughput against retrieval precision, so it is a P9 experiment, not a settled choice.
- **No quality claim is made about any model that was not benchmarked under the final
  configuration**, and **no pricing claim is made at all** — none was available.
- Document size skews dense retrieval: on a corpus capped at 40 chunks per document R@1
  was 0.75, against 0.67 on the full corpus where one book is 84% of the index. This is a
  property of the corpus, and it is why `DENSE_WEIGHT=0.7` is treated as a starting point
  awaiting P9 calibration rather than a tuned constant.

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
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | D1–D31: every decision with its measurement or mutation evidence |
| [`docs/eval_report.md`](docs/eval_report.md) | P9: full metrics and threshold calibration *(pending)* |

## Quick start

```bash
uv sync
cp .env.example .env      # add EMBEDDING_API_KEY, EMBEDDING_MODEL=Bge-m3
make check                # ruff + mypy + pytest, fully offline
make run                  # uvicorn --workers 1  (a single worker is required)
```

See [`plan.md`](plan.md) §7 for every tunable and §13 for the release checklist.
