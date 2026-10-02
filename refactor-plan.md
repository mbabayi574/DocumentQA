# P11 — refactor plan

**Scope:** `plan.md` §9 P11 ("Refactor and README"), audited against `jobTask.md`'s
requirements and §10's engineering rules, plus a repo-wide `ponytail-audit` pass.

**Gate, unchanged from §0.2:** every commit below needs `make check` green, **`make live`
green against the real BGE-M3**, and a fresh `git clone` green. `make live` is not
optional — four of the defects in `docs/DECISIONS.md` (D45–D48) were invisible to 480
offline tests, and R5 moves the hash function that I5 and I10 rest on.

**Audit method, so the coverage claim is honest:** all 43 modules under `src/qasystem`
(5 765 lines) and `tests/eval/runner.py` (1 202 lines) were read; the remaining
10 274 lines of tests were swept mechanically for unused definitions, uncalled public
methods, single-caller symbols, internal import edges, unused parameters, one-method
classes, delegation-only functions, bare `except`, bare `except: pass`, and
config-with-no-reader. Every finding below names the file and line it came from.

---

## 1. Audit result

Ranked biggest cut first. Tags: `delete:` dead code · `stdlib:` hand-rolled thing the
stdlib ships · `native:` a dependency doing what the platform does · `yagni:`
abstraction with one implementation · `shrink:` same logic, fewer lines.

| # | tag | finding | replacement | evidence |
|---|---|---|---|---|
| 1 | `delete:` | **The query LRU cache.** `CachingEmbedder.embed_query` + `DEFAULT_QUERY_LRU_SIZE` + the `query_lru_size` kwarg + `_query_lru` — a third cache in front of a cache that already answers the question. The query path calls `embed([question])`, never `embed_query`. | nothing; the persistent SQLite cache already makes a repeated question cost zero requests, proved live | `embeddings/caching.py:24,48–53,90–101`; `retrieval/service.py:128`; only callers are `tests/unit/test_embed_caching.py:100–123` |
| 2 | `delete:` | **`.qasystem.lock` is committed.** A 0-byte runtime lock file, tracked in git, because `.gitignore` covers only `/data/` and this one was created in the repo root. | `git rm --cached` + one `.gitignore` line | `git ls-files` → `.qasystem.lock`; `.gitignore:14–15` |
| 3 | `delete:` | **`docs/eval_report.md` does not exist.** 3 broken README links, 11 references across `plan.md`/`DECISIONS.md`/docstrings, one **ticked** DoD box, and `DECISIONS.md` contradicting itself (`:1442` says `tests/eval/report.md`, `:215` and `:1654` say `docs/eval_report.md`). The artifact is `tests/eval/report.md`, written by `cli.py:179`. | make the path real (R3) | `README.md:40,75,162`; `plan.md:105,346,635,771`; `DECISIONS.md:215,1442,1654`; `cli.py:179` |
| 4 | `delete:` | **`_Cache` in the ingestion tests** — a hand-rolled copy of production `SqliteEmbeddingCache`, in the file whose own docstring claims "so I5 and I10 use the real table". | `SqliteEmbeddingCache(store)`, as `test_retrieval.py:143` already does | `tests/integration/test_ingestion.py:179–189` vs `src/qasystem/storage/embedding_cache.py:17–27` |
| 5 | `yagni:` | **`reconcile.EmbedderRef`** — a one-method Protocol shadowing the existing `Embedder` port, existing only so a test can pass a tripwire. | `domain.ports.Embedder`, which already declares all three members | `ingestion/reconcile.py:25–32` vs `domain/ports.py:14–39` |
| 6 | `delete:` | **`storage/embedding_cache.py`** — a 27-line file whose entire body delegates to `SqliteStore.get_embeddings`/`put_embeddings`. Its docstring says so in the first sentence. | fold the two methods into `sqlite_store.py` under the port's names; the `EmbeddingCache` Protocol stays (the in-memory fake is a real second implementation) | `storage/embedding_cache.py` whole file; `deps.py:36,98` |
| 7 | `delete:` | **`sentence_rerank`** — a config flag with zero readers. D65 already records that `SENTENCE_RERANK` has no implementation. | nothing; §9.4's experiment 3 stays unbuilt until it is measured | `config.py:35`; `src` readers: 0 |
| 8 | `delete:` | **`_int_or_none`** — defined, never called. | nothing | `ingestion/service.py:412–416`; `src` callers: 0 |
| 9 | `shrink:` | **Three parameters nothing reads**: `_install_request_id(app, settings)`, `replace_document(..., request)`, `_join(self, question, ...)`. | drop the parameter | `api/app.py:121,75`; `api/routes.py:221`; `retrieval/service.py:245` |
| 10 | `shrink:` | **Three bad import edges** the AST purity test P11 asks for would fail on: `storage.sqlite_store → embeddings.caching` for a `sha256` one-liner; `ingestion.reconcile → storage.chroma_store` for the integer `PAGE_SIZE`; and an `answering.extractive ⇄ retrieval.service` cycle via `Candidate`. | `input_hash` and `Candidate` to `domain/models.py`; `reconcile` names its own batch size | `sqlite_store.py:28`; `reconcile.py:21`; `extractive.py:31` ⇄ `retrieval/service.py:39` |
| 11 | `delete:` | **`scripts/eval.sh` in `plan.md` §7's layout** — never existed. | delete the line | `plan.md:347`; `git ls-files scripts/` |
| 12 | `shrink:` | **`SqliteStore`'s docstring claims "a caller cannot reach a cursor"; `execute_script` hands one out.** The D72 class of defect — a docstring that is not true. | one sentence | `sqlite_store.py:4` vs `:122–124` |

**net: −96 lines, −1 file, −1 dead config flag, −3 bad import edges, 0 dependencies.**

The honest headline: **the architecture is lean and the fat is in the documentation.**
96 lines out of 16 039 is 0.6%. Items 2 and 3 — a committed lock file and a DoD box
ticked for a file that does not exist — are worth more to a reviewer than all of
items 1, 4–9 combined. P11's real work is R8 (the README) and R7 (the decisions index),
not the deletions.

---

## 2. Work, in commit order

Each item is one commit, test-first, `make check` + `make live` + clean clone green.

### R1 — Delete the dead cache, the dead flag, and the dead helpers
Findings 1, 7, 8, 9. 29 lines of source, 26 of tests.

**Test first.** Deletion needs a test that says the *surviving* property, not one that
says the deleted code. `test_a_repeated_question_costs_no_embedding_request`
(`tests/integration/test_retrieval_live.py:272`) already proves the property through the
real query path with real vectors — that is the test to point at, and the two LRU unit
tests go with the code they test. Then delete, and let the suite say what breaks.

**Why the LRU goes rather than being wired in.** `retrieve()` reaches the embedder
through `embed()`, which already consults the persistent cache keyed on
`sha256(exact embedded input)`. The LRU would only have saved one SQLite read: D63
measured the warm-cache path at **P50 10 ms**, which is the system's own share of a
query whose other half is a 354 ms provider call. A third cache in front of the second
is a thing to maintain, document, and test for a saving nobody would notice — and §10
rule 18 calls an option with no caller a bug waiting to be documented as a feature.

**Acceptance:** `grep -rn "embed_query\|sentence_rerank\|_int_or_none"` over `src/`
returns nothing; `make check` and `make live` green; the DECISIONS entry records that
the query path uses `embed()` and cites D63 for the cost of that choice.

### R2 — Untrack the runtime lock
Finding 2. One file, one line of `.gitignore`.

`.qasystem.lock` is L2's single-instance lock, an artifact of running the app. A
committed lock file invites a reader to reason about ownership of `data/` from a file
that means nothing in a fresh clone. Add `/.qasystem.lock` beside the `/data/` rule with
a one-line reason, `git rm --cached`, and extend `test_repo_consistency.py` — which
already guards fixtures against exactly this failure (D31) — to assert no `*.lock`
outside `uv.lock` is tracked.

### R3 — Make `docs/eval_report.md` exist
Finding 3. One line of code, one `git mv`, two references.

`EVAL_DIR` is the harness's location and the report is a deliverable; they were welded
together at `cli.py:179`, which is why the report landed in `tests/` while
`plan.md` §7, §9's P9 gate and §11's ticked box all say `docs/eval_report.md`.
`AGENTS.md` makes `plan.md` the source of truth, so the documentation is right and the
code drifted: add a `REPORT_PATH` constant beside `EVAL_DIR`, write there, `git mv` the
file, and update the two readers (`README.md:174`, `test_eval_live.py:101`).

*Alternative considered and rejected:* repoint four documentation references at
`tests/eval/report.md`. Cheaper by one line, and it makes §11's ticked box false and
parks a generated deliverable inside the test tree. One line of code is the smaller
truth.

**Guard, because D31 has now happened twice.** Extend `test_repo_consistency.py` with a
markdown-link check over `README.md`, `docs/DECISIONS.md` and `plan.md`: every relative
`](…)` target must exist. That is 12 lines and it closes the whole class — the current
test says outright that it does not try to check path references.

### R4 — Remove the duplicate port, the duplicate test class, and the delegate file
Findings 4, 5, 6. ~40 lines, −1 file.

* `EmbedderRef` → the existing `Embedder` port. The tripwire in
  `test_ingestion_reconcile.py` gains a `requests` attribute; that is the honest cost of
  typing the argument against the real contract instead of a one-method shadow of it.
* `_Cache` → `SqliteEmbeddingCache`. This is not only a deletion: it means the tests
  that certify I5 and I10 certify them against the **production adapter**, which the
  copy could not have done if the two ever diverged.
* `embedding_cache.py` → deleted, and the `EmbeddingCache` Protocol in
  `embeddings/caching.py` renames its two methods to `get_embeddings`/`put_embeddings`,
  which are the names `SqliteStore` already has. `SqliteStore` then satisfies the port
  structurally and `deps.py` stops constructing an adapter that only delegates.
  *Direction matters:* renaming the store's methods instead would touch 16 call sites in
  `test_store_sqlite.py` and `reconcile.py:76`; renaming the port's touches 3, and
  deleting `_Cache` (above) removes one of them. The Protocol itself stays — the
  in-memory fakes in `test_embed_caching.py` and `test_retrieval.py` are real second
  implementations, and §10 rule 16 requires one.

**`sqlite_store.py` stays at ~610 lines and is not split.** D29 argued this for the
559-line version and nothing has changed the argument: a split needs either a base class
plus a second connection to the same file, or a public `connection` handle, and
`test_the_raw_connection_is_not_leaked_to_callers` exists to keep the second out. The
plan's "roughly 400" was written before the code existed and was a guess; P11 records
the measured number and the reason instead of reorganising a cohesive repository to
satisfy it. (`sqlite_store.py` is 637 lines today and lands near 610.)

### R5 — Fix the three bad import edges
Finding 10. Net ≈ +6 lines, −3 edges.

Measured graph (`src` → `src`), the rules the extended AST test will enforce:

| rule | today | after |
|---|---|---|
| `domain/` and `text/` import nothing outside themselves | holds | holds |
| `storage/` must not import `embeddings/` | **violated** (`sqlite_store.py:28`) | holds |
| `ingestion/` must not import a concrete adapter (`storage.chroma_store`) | **violated** (`reconcile.py:21`) | holds |
| `answering/` must not import `retrieval/` | **violated** (cycle, `extractive.py:31`) | holds |

* `input_hash` → `domain/models.py`. One definition, no inversion, and crucially no
  *duplicated* hash: `sqlite_store` writes `embed_input_hash` and `caching` reads it back
  as the cache key, so two copies of the expression would break I5 and I10 silently the
  day one of them changed. This is the one edit in P11 that touches the hash, which is
  why R5's live run is not optional.
* `reconcile` names its own batch size. `PAGE_SIZE` is Chroma's page size (D29's
  `get(limit=, offset=)` finding); `reconcile` happens to want the same integer for a
  different job. Two named constants with equal values is honest; one shared constant
  across a layer boundary is not.
* `Candidate` → `domain/models.py`, beside `Chunk` and `VectorHit`. It is the answerer's
  *input* type, so `answering` depending on `retrieval` for it is backwards, and the
  `TYPE_CHECKING` import is what has been hiding the cycle from mypy's cycle detection
  rather than from a reader.

**Acceptance:** `tests/unit/test_domain_purity.py` grows the three rules above, and the
new tests are written **first** and watched to fail on today's code — a layer test added
after the fix proves nothing (D25, D26, D44, D56 are all the same lesson).

### R6 — Group the invariants behind `pytest -m invariant`
`plan.md` §9 P11. Today: 0 markers, and `--strict-markers` is on, so the marker does not
exist yet.

Declare `invariant` in `pyproject.toml` and tag the named witness for each of I1–I10, so
the release blockers are verifiable without reading the suite:

| | invariant | witness to tag |
|---|---|---|
| I1 | active version only | `test_a_published_chunk_is_eligible`, `test_a_superseded_chunk_is_not_eligible`, `test_a_staging_chunk_is_not_eligible` |
| I2 | deletion, cleanup fault-injected | `test_5_delete_is_unretrievable_even_when_chroma_cleanup_fails` |
| I3 | atomic publication | `test_7_embedder_failure_mid_update_leaves_the_old_version_queryable`, `test_7b_a_failed_version_is_marked_and_leaves_no_rows`, `test_8_chroma_failure_during_staging_aborts_the_publish`, `test_a_rolled_back_staging_can_be_retried` |
| I4 | idempotency | `test_2_identical_reupload_is_unchanged_and_costs_zero_embed_calls`, `test_2b_different_bytes_with_identical_parsed_text_are_also_unchanged` |
| I5 | unchanged chunks cost zero requests | `test_3_a_local_edit_embeds_only_the_changed_chunk`, `test_a_miss_embeds_and_writes_back` |
| I6 | traceability | `test_3_every_segment_is_an_exact_substring_of_its_chunk_and_the_source` |
| I7 | no synthesis | `test_3b_the_rendered_answer_is_exactly_the_slices_plus_their_markers`, `test_3c_segments_are_ordered_by_source_position` |
| I8 | secrets | the redaction/scrub tests plus `test_the_token_never_appears_in_an_error_body` |
| I9 | model isolation | `test_a_different_dimension_gets_a_different_collection`, `test_a_name_collision_is_refused_rather_than_mixed` |
| I10 | disposable index | `test_10_rebuild_restores_dense_search_with_zero_embed_calls` |

Add `make invariants` → `pytest -m invariant -q`, and record the count it prints. A
number the reviewer cannot check is not a grouping.

### R7 — `docs/DECISIONS.md` as the reviewer's entry point
`plan.md` §9 P11. 72 entries, chronological, "newest last" — good for the author, wrong
for a reviewer, who wants "where is the model rationale" in one read.

Keep the chronological log untouched (its order *is* the evidence: each entry corrects
the one before) and add an index above it that answers the six questions a reviewer
actually arrives with:

1. **Why BGE-M3 and not the other three** → README §3, then D59, D71.
2. **Why these thresholds** → D59 (weights), D62 (medoid tie-break), D67 (`min_lexical`
   removed), D71 (`min_dense_alone` removed), and `config/thresholds.json`'s own notes.
3. **Why this chunk size** → D17 (sized on the densest measured text), D64 (the
   experiment that could not discriminate).
4. **Every deviation from `plan.md`** → D8, D11, D14, D18, D20, D21, D28, D36, D41,
   D47–D48, D51–D53, D57–D71 — one line each, with the plan section it departs from.
5. **The defects that were invisible offline** → D45–D48, D49, D50.
6. **The tests that could have passed for the wrong reason** → D25, D26, D44, D56, and
   the three mutation tables.

Add one `jobTask.md` requirement → decision mapping, since that is the reviewer's actual
reading order. ~60 lines of index, no rewriting of history.

### R8 — The README, against §9's twelve items
`README.md` is 179 lines and currently covers **three** of the twelve. This is the
largest single piece of P11 and the one a reviewer reads first.

| # | item | state | action |
|---|---|---|---|
| 1 | problem, constraints, A1 | **missing** | write |
| 2 | architecture, why no LLM generation | **missing** | §4's diagram + "the provider has no chat endpoint" |
| 3 | **BGE-M3 and why** | **done**, 140 lines | add D71's correction |
| 4 | storage design, backup, `rebuild` | **missing** | write |
| 5 | version publication, `eligible_chunks` | **missing** | write |
| 6 | chunking, Persian normalisation | **missing** (`:147` is about the provider's `index` field — D21, not chunking) | write: `CHARS_PER_TOKEN=1.5` on the densest measured text (D17), no cross-section merging (D18), the mechanical cap pass (D19), ZWNJ's three forms (D8), NFKC in the PDF parser only (D14), and the Persian book that loses 274 of 6 563 spaces (D16) |
| 7 | retrieval, gate, answer selection, `evidence_score` | **missing** | write; state that `evidence_score` is a deterministic score, not a probability |
| 8 | eval dataset, metrics, calibration, limits | partial (`:35` measured quality) | add the dev/held-out split table, `gold quoted` as the acceptance metric for requirement 4 (D66), and D64's "no change adopted because nothing was measured" |
| 9 | **`curl` examples** | **missing** | write for upload/update/delete/query |
| 10 | setup, single worker, tests | partial (`:164`) | add `make invariants` |
| 11 | known limitations | partial (`:112`) | add no-OCR, single-process, small eval set, the 84% skew, PyMuPDF licence |
| 12 | security: token externalised, redacted, never committed | **missing** | write |

Two things the README must say that it currently does not, both because the user asked
for them and both measured:

* **Queries are English or Persian, and cross-lingual retrieval is not supported.** D71
  removed the branch that carried it, `config/thresholds.json` records the removal in
  its own notes, and `plan.md` §2's headline claim is now false. The README already
  gestures at this; it must say it plainly, with the D61 measurement behind it.
* **The Swagger UI at `/docs`.** One line in Quick start is already there; §9 gets the
  `curl` equivalents, because a reviewer without the page open needs them.

### R9 — The consistency pass
`plan.md` §9 P11. One pass over error codes, config names, docstrings and `plan.md` §7,
after R1–R8 land, so it checks the final state:

* every `code` in `errors.py` appears in the OpenAPI schema or is documented as
  internal; the two literals with no class behind them (`VALIDATION_ERROR`,
  `INTERNAL_ERROR`) get a comment saying why;
* every field in `Settings` has a reader in `src` or is deleted (this is what caught
  `sentence_rerank`);
* `plan.md` §7's layout matches `git ls-files` (this is what caught `scripts/eval.sh`);
* `sqlite_store.py`'s docstring stops claiming no caller can reach a cursor.

---

## 3. Verified *not* to change

Recorded so this audit is not repeated, and so nobody "fixes" a deliberate choice.

| checked | measurement | verdict |
|---|---|---|
| coverage on §11's six directories | 93% overall; lowest is `answering/sentences.py` at 90% | gate is ≥85% — **met, no work** |
| magic numbers inline (§10 rule 13) | every float literal outside `config.py` is `0.0`/`1.0`/`0.5` arithmetic or a `§7.4` docstring reference | **clean, no work** |
| error-code consistency | one envelope; `VALIDATION_ERROR`/`INTERNAL_ERROR` are the only class-less literals, and both are deliberate — FastAPI's validation error has no domain class | **clean** |
| the duplicated `env` fixtures in 4 integration test files | they differ per layer (`FlakyVectors`, a `retriever()` override, a live env); collapsing them needs a parameterised factory | **leave** — more complexity than the duplication |
| the 8 test-only `SqliteStore` accessors (`lexical_count`, `version_state`, `source_text`, `ingest_log`, `cached_dimensions`, `journal_mode`, `foreign_keys_enabled`, `in_transaction`) | they are the mechanism by which I1–I10 are mechanically proven | **leave** — deleting moves SQL into tests, which is worse |
| `sqlite_store.py` at 637 lines vs the plan's ~400 | D29's argument is unchanged; a split needs a base class or a leaked connection | **leave, and record why** |
| `embeddings/fake.py`, `rate_limit.py`, `scripts/measure_provider.py` | each has one caller and is load-bearing (offline determinism, the 120/min limiter, §2's measurements) | **leave** |
| `VectorStore`'s 8 methods | all eight are called by ingestion, reconcile or the live tests | **leave** |
| `EmbeddingCache` Protocol | two real implementations (SQLite adapter, in-memory fake) | **leave** — §10 rule 16 |
| `build_pdf_fixtures.py` / `build_eval_corpus.py` | never re-run; they are the only record of how two committed artifacts were produced | **leave**, and add `build_eval_corpus.py` a reference in the README, which currently has none |

---

## 4. Two questions for the human

1. **`docs/eval_report.md` vs repointing the docs (R3).** Recommendation: make the file
   real, because `plan.md` is the source of truth and §11's box is already ticked. The
   alternative is four doc edits and one permanently false tick.
2. **§11's unticked boxes.** Eight of the twelve DoD boxes are unticked although the
   tests exist (PDF/TXT/MD ingestion, add/update/delete with zero-request re-upload,
   chunk-level re-embedding, I1–I3 with fault injection, I3 partial versions, Chroma
   persistence and single-instance lock, exact-excerpt answers with mechanical citation
   verification, no GUI / stable OpenAPI / no secrets). Ticking them is a claim that
   each has a named test, so R6 is what makes it honest — the tick should cite the
   marker, not the prose. One box is a judgement call: `gold quoted` is 0.913 dev /
   0.875 held-out against no target number, because D66 set the target as "remove the
   causes" rather than "pass a bar".

---

## 5. Gate checklist for P11

- [ ] R1–R9 landed as separate commits, each with `make check` **and** `make live` green
- [ ] `pytest -m invariant` runs and its count is recorded here
- [ ] AST purity test extended past `domain/`, written red first
- [ ] `README.md` covers all twelve §9 items, and says cross-lingual is unsupported
- [ ] `docs/DECISIONS.md` opens with an index; the chronological log is unchanged
- [ ] Every relative markdown link in `README.md`, `DECISIONS.md` and `plan.md` resolves
      (guard test in `test_repo_consistency.py`)
- [ ] No `*.lock` tracked outside `uv.lock`
- [ ] `grep -rn "embed_query\|sentence_rerank\|_int_or_none\|EmbedderRef"` over `src/`
      returns nothing
- [ ] `plan.md` §7's layout matches `git ls-files`
- [ ] Clean clone: `uv sync && cp .env.example .env && make check && make run`, then
      `make live`