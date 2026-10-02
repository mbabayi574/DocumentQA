# Evaluation report — BGE-M3 (plan.md §9)

Model `Bge-m3`, 1024 dimensions, vectors L2-normalized by
the provider so ingest does no normalization of its own (§2).

- corpus: **6 documents**, 49 chunks, 11,285 characters
- index: **49 vectors**; ingest cost **0 provider request(s)**
- questions: **50** (30 dev / 20 test), split by a rule fixed before any result was seen
- embedder requests total for this run: 1

## Retrieval and gating: three systems, all questions (plan.md §9.2)

| configuration | answerable | unanswerable | R@1 | R@3 | R@5 | MRR@5 | answered | false answers | gold quoted |
| dense-only | 38 | 12 | 0.68 | 0.95 | 0.95 | 0.807 | 0.92 | 0.00 | 0.92 |
| lexical-only | 38 | 12 | 0.58 | 0.76 | 0.87 | 0.687 | 0.89 | 0.00 | 0.89 |
| hybrid | 38 | 12 | 0.76 | 0.95 | 0.95 | 0.846 | 0.92 | 0.00 | 0.90 |

R@k is how many candidates it takes before **every** gold phrase is in the window, so
a two-fact question is not scored as answered on one of its facts. Gold is a phrase
from the source, never a chunk id (§9.1).

**gold quoted** is the share of gold phrases that appear in a chunk the answer actually
cites, over the answerable questions that were answered. It is the only column that
catches an *answered question citing the wrong document*, which R@k cannot see (R@k is
about the window) and the false-answer rate cannot see (that column is unanswerable
questions only). A citation asserts that its source supports the answer, so this is the
column requirement 4 of jobTask.md is actually about (D60).

## Per split (hybrid)

| configuration | answerable | unanswerable | R@1 | R@3 | R@5 | MRR@5 | answered | false answers | gold quoted |
| dev | 23 | 7 | 0.83 | 0.96 | 0.96 | 0.884 | 0.91 | 0.00 | 0.91 |
| test | 15 | 5 | 0.67 | 0.93 | 0.93 | 0.789 | 0.93 | 0.00 | 0.88 |

## Latency

- end to end, each question asked once, cold: **P50 358 ms, P95 1385 ms**, max 2847 ms
- the same questions again with the query cache warm, so no network: **P50 10 ms**, P95 11 ms

The tail is the embedding provider's, not this system's (D54). Everything here builds
is a small fraction of one network round trip, so a P95 quoted without this
qualification is measuring someone else's server.

## Mechanical checks

- **citation substring validity: 103 segments verified, 0 failures.** Each
  segment equals its chunk's `text[start:end]` *and* the version's
  `source_text[char_start + start : char_start + end]`, and the excerpt contains it,
  which is I6 and I7 (plan.md §5.1).
- **stale-content leakage: 0.** Measured, not assumed: a document was
  republished with a phrase removed and all 50 questions were re-asked. The old text was still in FTS, in the
  superseded version and in the vector store for the whole of that window.

## The evidence gate

```json
{
  "min_dense": 0.54,
  "min_coverage": 0.4,
  "min_coverage_high": 0.75,
  "min_sentence_overlap": 0.15
}
```

Chosen by grid search on **dev** only: 2088 feasible of 4641 tested (a point is feasible when it refuses every
unanswerable dev case). Reported on the held-out split by rebuilding the real
service with these numbers.

| split | answerable | answered | unanswerable | false answers |
|---|---|---|---|---|
| dev | 23 | 0.91 | 7 | 0 |
| test (held out) | 15 | 0.93 | 5 | 0 |

§9.3's target is a false-answer rate within 5% on both
splits, preferring 0%, and it is met on both. The first live run on the fixture
corpus measured **1 in 9** (D47), and the reason that does not reproduce here is
D69 and D71: the uncorroborated branch that carried the false answers has been
removed, and re-calibrating without it left every metric above **identical**. The
2 088 feasible points are now chosen among thresholds where *every* path into the
gate requires coverage, so a question whose evidence shares no token with it is
refused however close the embedding. That is also why cross-lingual retrieval is
**not supported**: retrieval still finds the cross-language chunk (R@3 is 0.95), but
nothing the system can measure admits it, and D61 measured the two cross-lingual
answers that did get through as lexical coincidence rather than the embedding.

## Answered, but citing a source that does not contain the answer

The failure `gold quoted` measures, named case by case. Every one of these was
**answered** — the gate passed and citations were produced — and not one of them is
an unanswerable question, so §9.3's false-answer rate is 0.00 for all of them. Each
either cites a document that does not hold the answer, or cites the right document
and omits the fact from the quoted slice. Both mislead a reader who cannot check,
and a caller who cannot read the cited language has no way to notice at all (D60).

| case | split | question | cited | gold not quoted (cause) |
|---|---|---|---|---|
| `fa02` | dev | گیت‌وری با چه دستوری نصب می‌شود؟ | deploy-guide.md | `auroractl install --channel stable` (truncation, sentence overlap 0.000) |
| `x01` | dev | What is the exact path of the file that holds the gateway's configuration? | handbook.md | `/etc/aurora/gateway.toml` (retrieval) |
| `m01` | test | What port does the gateway listen on, and what sustained throughput does one tenant get? | handbook.md, limits.pdf | `0.0.0.0:8443` (answerer, sentence overlap 0.067) |
| `m05` | test | What are the installation prerequisites and the default listen port? | handbook.md, limits.pdf | `۴ هسته` (retrieval, sentence overlap 0.000) |

Three causes, and the `kind` is the useful part — they need different fixes. The
sentence overlap is the answerer's own score for the sentence holding the fact, so
whether a case has a *second* blocker is visible rather than assumed.

* **retrieval** — the citation names a document that does not hold the answer. This
  is D55's mechanism on a corpus where every question has a known answer: the dense
  arm finds the answer (R@3 is 0.95) and the lexical arm's incidental word matches
  put a same-language document in front of it.
* **truncation** — the right document *is* cited and the gold chunk is in the
  candidate window, but below `top_k`, so the answerer never saw it. The obvious
  fix is `TOP_K`, and P10 measured it: `gold quoted` moves 0.897 -> 0.923 between
  `top_k` 5 and 15 for +19% answer length, and the truncation case *converts* to an
  answerer case rather than being fixed. `TOP_K` is therefore left at 5.
* **answerer** — the gold chunk is inside `top_k` and the sentence carrying the fact
  was dropped by `min_sentence_overlap`. Every non-retrieval case above is this one
  defect, and it is a limitation of the signal rather than a tuning miss. `m01` asks
  "what port" and the sentence holding `0.0.0.0:8443` scores **0.067** against a
  0.15 bar, while the sentence beside it — which talks *about* the port without
  containing the number — scores **0.153** and is quoted instead. `fa02` asks
  "which command" and the gold sentence is a Markdown code fence at **0.000**: the
  command shares no word with the question, and the word that connects them is in
  the section heading, not the sentence. Word overlap cannot connect a question in
  words to an answer
  given as a literal, and lowering the bar to reach these would pad every answer with
  unrelated sentences, which §7.3 forbids. This is the one place a dense re-score of
  candidate sentences would help — which is what `SENTENCE_RERANK` was for, and why
  §9.4 measured it and found nothing to measure: there was no implementation.

Neither is a false answer in §9.3's sense — every case here is an answerable
question that the gate was right to admit. They are the residue that a false-answer
rate of 0.00 does not see, and `gold quoted` is the number that does.

## §9.4 experiments

### 1. Fusion weights

| dense | lexical | R@1 | R@3 | R@5 | MRR@5 | answered |
|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 0.58 | 0.76 | 0.87 | 0.687 | 0.89 |
| 0.10 | 0.90 | 0.58 | 0.79 | 0.89 | 0.699 | 0.92 |
| 0.20 | 0.80 | 0.63 | 0.82 | 0.89 | 0.736 | 0.92 |
| 0.30 | 0.70 | 0.66 | 0.84 | 0.89 | 0.762 | 0.92 |
| 0.40 | 0.60 | 0.68 | 0.87 | 0.92 | 0.782 | 0.92 |
| 0.50 | 0.50 | 0.68 | 0.87 | 0.92 | 0.784 | 0.92 |
| 0.60 | 0.40 | 0.71 | 0.89 | 0.92 | 0.804 | 0.92 |
| 0.70 | 0.30 | 0.74 | 0.89 | 0.92 | 0.817 | 0.92 |
| 0.80 | 0.20 | 0.74 | 0.89 | 0.92 | 0.821 | 0.92 |
| 0.90 | 0.10 | 0.76 | 0.95 | 0.95 | 0.846 | 0.92 |
| 1.00 | 0.00 | 0.68 | 0.95 | 0.95 | 0.807 | 0.92 |

Best by R@1: **0.90/0.10**. Shipped: **0.90/0.10**.

### 2. Chunk size

| target / hard max (tokens) | approx chars | chunks | ingest requests | R@1 | R@3 | MRR@5 |
|---|---|---|---|---|---|---|
| 350 / 700 | 525 | 49 | 6 | 0.76 | 0.95 | 0.846 |
| 1000 / 2000 | 1500 | 49 | 6 | 0.76 | 0.95 | 0.846 |
| 4000 / 8000 | 6000 | 49 | 6 | 0.76 | 0.95 | 0.846 |

**This experiment cannot discriminate on this corpus, and the reason is measured:**
the corpus is 49 sections in 49 chunks, and its longest
section is **635 characters**. The operative knob is the hard cap,
not the packing target: a section is at least one chunk, and it only becomes two when
it does not fit inside the hard cap. The smallest hard cap tried here is
**1050 characters**, and no section in this corpus reaches it,
so every row is 49 chunks by construction rather than by measurement.

A grid that stopped at 1000 tokens would have reported "chunk size does not matter"
— a conclusion drawn from a search space that could not have found one (D57). Making
this experiment real needs a corpus with a section longer than the hard cap, which is
a corpus change and not something to smuggle in by inflating a grid.

**No change is adopted, because nothing was measured.** The cost half of the trade-off
is §2's table: cost per character falls 9.5x from 500- to 8000-char items. Adopting a
larger chunk on the strength of that alone would be trading citation granularity for
throughput on no retrieval evidence at all, so the question stays open in §12.

### 3. Sentence reranking and 4. the per-item cap

Neither was run, deliberately. `SENTENCE_RERANK` has no implementation, so there is
nothing to measure and building a reranker before knowing whether sentence selection
is a measured problem is the speculative work §10 rule 16 forbids. The per-item cap
cannot affect retrieval quality at all: `MAX_CHARS_PER_ITEM` is a provider guard, the
chunker produces chunks an order of magnitude below it (§2 measures a 40 949-char
ceiling against a 1 050-char hard cap), and lowering it would only convert a request
that succeeds into one that fails. Neither is an experiment; both are arguments.
