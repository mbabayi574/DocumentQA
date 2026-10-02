# Decisions

Log with evidence. Newest last.

## D1 — package renamed `src/documentqa` → `src/qasystem` (P0)

**Context:** `uv init` scaffolded `src/documentqa/`; plan.md §7 names `src/qasystem/`.
**Decision:** follow plan.md. `src/documentqa/__init__.py` contained only a `uv`
hello-world stub, so nothing of value was lost. `pyproject.toml` name is now `qasystem`.
**Evidence:** plan.md §3/§4/§7 all reference `qasystem.*` import paths; keeping two
package names would have forced a rename later with tests importing the wrong one.

## D2 — `requires-python = ">=3.12"`, not plan.md's `>=3.11` (P0)

**Context:** plan.md §7 P0 says "Python ≥ 3.11". The pinned `numpy==2.5.3`
requires `>=3.12`, and `chromadb==1.5.9` pulls `onnxruntime`/`kubernetes`, which
are also 3.12+. The local venv runs 3.13.
**Decision:** set `>=3.12` and keep the pins. Verified by `uv sync`:
> `the requested Python version (>=3.11) does not satisfy Python>=3.12 and numpy==2.5.3 depends on Python>=3.12`
**Consequence:** a 3.11-only deployment is not supported. Nothing in the plan needs it.
`ruff`/`mypy` targets moved to `py312` to match.

## D3 — `EMBEDDING_MODEL` is required, never defaulted (P0)

**Context:** constraint C4 forbids hard-coding model IDs ("casing may differ, e.g.
`Bge-m3`") and requires discovering exact IDs from `GET /v1/models` in P4. The
provider's live ID in `.env` is `Bge-m3`, which is not the string in
`jobTask.md` ("BGE-M3").
**Decision:** `embedding_model` has no default; `load_settings` raises
`ConfigError` when `EMBEDDING_PROVIDER=remote` and it is missing. `.env.example`
leaves it empty with a comment. This makes a wrong-case guess impossible to ship.
**Rejected:** defaulting to `"BGE-M3"` — it would look valid in config and fail
only at the first real API call in P4, with a worse error.

## D4 — provider limits are enforced in config, not only documented (P0)

**Context:** C5 gives 120 req/min and 200 000 chars/request. §6's example config
says `RATE_LIMIT_PER_MIN=100` "stay under the 120/min provider limit" — a default,
not a ceiling.
**Decision:** `MAX_CHARS_PER_REQUEST` and `RATE_LIMIT_PER_MIN` are validated against
the provider maxima (`PROVIDER_MAX_CHARS_PER_REQUEST`, `PROVIDER_MAX_REQUESTS_PER_MIN`);
values above them are a `ConfigError` naming the offending variable. Same for
`MAX_UPLOAD_MB > 0`, `CHUNK_HARD_MAX_TOKENS >= CHUNK_TARGET_TOKENS`,
`CHUNK_OVERLAP_RATIO ∈ [0,1)`, `EMBEDDING_DIMENSION > 0` when set.
**Reason:** a misconfigured limit that only fails at the API is a support ticket;
one that fails at startup with a named field is a log line.

## D5 — `EMBEDDING_PROVIDER=fake` allowed in `test` and `demo` only (P0)

**Context:** plan.md §6 says `fake` is "only when `APP_ENV in {test, demo}`;
shown in /ready". §4/test list requires `fake` rejected in `prod`.
**Decision:** `FAKE_PROVIDER_ENVS = {"test", "demo"}`; anything else (incl. `dev`)
raises `ConfigError` mentioning both `EMBEDDING_PROVIDER=fake` and `APP_ENV`.
**Note for P8:** `Settings.is_fake_provider` is the flag `/ready` must surface.

## D6 — redaction filter scrubs only `str` args (P0)

**Context:** `SecretRedactionFilter` initially rewrote every logging arg via
`str()`. Booting the app printed two logging tracebacks:
> `TypeError: %d format: a real number is required, not str`
because httpx logs `HTTP Request: %s %s "%s %d %s"` with an `int` status code, and
str-ifying it broke the `%d` spec.
**Decision:** scrub args only when they are already `str`; pass other types through
untouched. Regression test `test_redaction_preserves_non_string_args` reproduces the
exact httpx format string.
**Rule learned:** a logging filter that mutates `record.args` must preserve types,
otherwise it silently breaks unrelated loggers.

## D7 — the `test_*` list drives the P0 file set (P0)

Built exactly the modules the P0 "Tests first" list names, no more:
`config.py`, `errors.py`, `logging_setup.py`, `domain/models.py`, `domain/ports.py`,
`embeddings/fake.py`, `api/app.py` (liveness only), plus `Makefile`, `.env.example`.
`api/routes.py`, `schemas.py`, `deps.py`, `cli.py` and `config/thresholds.json` are
P8/P9 artifacts; the Makefile already wires `smoke`/`eval`/`calibrate`/`rebuild` to
them so those targets need no rework.
## D8 — the ZWNJ triple, because one spelling is not enough (P1)

**Context:** a Persian document may write `فرمت‌های` (with ZWNJ) while a user
types `فرمت های` (with a space). plan.md P1 requires keeping ZWNJ compounds *and*
emitting split components "for recall". Emitting only the parts loses the compound;
emitting only the compound loses the spaced query.
**Decision:** `tokenize` emits three forms for a ZWNJ token — the compound
(`فرمت‌های`), the fused form (`فرمتهای`), and the parts (`فرمت`, `های`) — and
`normalize_for_index` keeps ZWNJ while dropping other zero-width characters.
**Evidence:** measured overlap between the two spellings is now
`{فرمت, های, pdf, markdown}`, so both directions match. A single-form tokenizer
matched on at most one side.
**Rejected:** deleting ZWNJ in normalization. It fuses `می‌رود` into `میرود`, which
silently changes the word and corrupts any stored offset comparison.

## D9 — Persian stopwords live in one string, with an explicit noqa (P1)

**Context:** `ruff`'s RUF001 flags Persian letters as ASCII-confusable (`ه` looks like
`o`, `ا` like `l`) and its formatter exploded the stopword list to one word per line,
which is unreadable for a vocabulary.
**Decision:** keep each language's stopwords in one space-separated string and add
`# ruff: noqa: RUF001` with the reason. List form is the formatter's preference, not a
readability win here.
**Lesson:** when a linter's fix is worse than the code, scope the rule out and say why
in a comment, rather than fighting it per line.

## D10 — TXT and Markdown sections are trimmed; PDF sections are pages (P2)

**Context:** a blank-line-separated block in TXT spans its trailing newline, and a
Markdown block spans its trailing blank lines. Both produce citation excerpts ending
in whitespace.
**Decision:** trim surrounding whitespace from every section's char span. The slice is
still a verbatim slice of `text` (I6 holds), it just does not quote the blank line.
**Evidence:** `test_crlf_is_normalized_to_lf` expects `"First."`, not `"First.\n"`.

## D11 — Markdown `hr` tokens are skipped, fenced code is kept (P2)

**Context:** parsing `storyen.md` produced 77 sections of which **10 were `---`**.
**Decision:** skip `hr` block tokens; keep `fence` blocks, because a fenced YAML or
shell snippet is genuine content a user may search for.
**Evidence:** sections dropped 77 → 67 on `storyen.md` with no prose lost. A section
made only of `---` would otherwise become a junk chunk with no retrieval value.

## D12 — PDF sections are per page, and no OCR, no RTL reversal (P2)

**Context:** a PDF's heading structure is not reliably recoverable without font
heuristics; and `justforfun_book_a4.pdf` (plan.md §2.3) stores Persian in lossy
presentation forms whose order no reversal or NFKC pass can restore.
**Decision:** one section per page that carries text, breadcrumb `<title> > page N`
with exact 1-based page numbers. A PDF where no page yields text raises
`NoTextLayerError` whose message says OCR is unsupported. **No RTL reversal anywhere.**
**Evidence:** `ai-engineer.pdf` yields 7 text-bearing sections across 8 pages — page 1
is blank and correctly skipped without voiding the document
(`test_blank_page_does_not_void_the_document`). Persian extracts correctly
(`test_persian_pdf_text_is_readable_not_reversed`).
**Rejected:** font-size heading detection. It is a heuristic on a fixture set of two
PDFs; a wrong heading is a wrong citation path, which is worse than page granularity.

## D13 — RUF001 ignored project-wide, with a reason (P2)

**Context:** P1 scoped the rule out inline on the stopword list. P2 then hit it again in
Persian test literals, so the inline approach would have grown a `noqa` on every line
containing Persian.
**Decision:** `ignore = ["RUF001"]` in `pyproject.toml`, with the reason in a comment:
this project is bilingual by requirement, so Arabic/Persian letters are content, not
homoglyph attacks. The security property that actually matters — that confusable input
cannot forge a different query — is covered by the normalization and tokenizer tests,
not by this lint rule.

## D14 — the PDF parser applies NFKC (P2, after a full scan corrected D12)

**Context:** D12 called `justforfun_book_a4.pdf` unrecoverable. That verdict came from
reading its **raw** extraction (`ﻣﻘﺪﻣﻪ`, Arabic presentation forms) on two pages. A full
page-by-page scan of all three PDFs contradicted it:

| PDF | Pages | Text chars | Presentation-form chars | Verdict after scan |
|---|---|---|---|---|
| `justforfun_book_a4.pdf` | 204 | 390 139 | 233 183 | **recoverable via NFKC** |
| `ai-engineer.pdf` | 8 | 2 240 | 0 | clean already |
| `Clean Code….pdf` | 10 MB, 312 | 347 893 | 0 | clean, zero mojibake |

Ordering was the other thing I had assumed wrong. A correct Persian line *ends* with
punctuation; a reversed one *starts* with it. After NFKC, **980 of 1 053 content lines
end with punctuation and only 73 start with it**, so the text is in correct reading order
and merely needs compatibility decomposition — it was never character-reversed.
**Decision:** the PDF parser applies NFKC to each extracted block.
**Evidence:** byte-for-byte a **no-op** on `ai-engineer.pdf` (`test_nfkc_does_not_alter_a_clean_text_layer`),
and the difference between unreadable and readable Persian on `justforfun`
(`test_persian_book_is_readable_after_extraction` finds `تفریح` and asserts zero
presentation-form codepoints survive).
**Why NFKC is safe for I6 here:** a PDF's `ParsedDocument.text` is *constructed* by us,
not uploaded bytes. NFKC maps presentation forms onto the letters a reader actually sees,
so it is closer to the rendered page than the raw codepoints. Markdown and TXT text is
still stored byte-for-byte, because there the user's bytes are the ground truth.
**Superseded part of D12:** "no normalization recovers it" was wrong.

## D15 — the 10 MB book is git-ignored; a 12-page excerpt is committed instead (P2)

**Context:** the English book parses perfectly (348 204 chars) but is 10 MB and 312
pages — too slow to parse in a test suite and too large to commit.
**Decision:** keep the source git-ignored and commit
`tests/fixtures/docs/en/clean-code-excerpt.pdf` (197 KB, 12 prose-dense pages, 35 886
chars), produced once by `scripts/build_pdf_fixtures.py`.
**Reason:** the fixture carries the same text layer for test purposes without the size.
The original stays on disk, and `test_the_uncommitted_book_still_parses_when_present`
asserts it still parses when a local copy exists.
**Also:** `test_parsing_pdf_fixtures.py` parametrizes over every PDF under
`tests/fixtures/docs/`, so the next fixture dropped in is covered without editing a test.

## D16 — known ceiling: `justforfun` loses spaces on 274 of 6 563 lines (P2)

**Context:** that book's font maps the space glyph to nothing on some lines, so they
arrive as one run of letters (`فقطبرایتفریح` for `فقط برای تفریح`).
**Decision:** accept it and state it. `test_known_limitation_some_lines_lose_their_spaces`
pins it at under 20% of lines, so a regression that made it worse fails the suite.
**Rejected:** Persian word segmentation to re-insert spaces. It needs a language model or
a dictionary, would guess at boundaries, and a wrong space is a wrong token that quietly
degrades retrieval — a larger risk than the one it removes.

## D17 — chunk size is planned on the densest measured text, not prose (P3)

**Context:** no tokenizer is available offline, so chunk size must be estimated in
characters. Measuring BGE-M3's `usage.prompt_tokens` gave very different densities:

| Sample | chars | tokens | chars/token |
|---|---|---|---|
| English prose | 232 | 56 | **4.14** |
| Persian prose | 172 | 40 | **4.30** |
| Persian with ZWNJ | 67 | 17 | 3.94 |
| Identifiers (`ERR-404`, `EMBEDDING_AUTH`, `hnsw:space`) | 116 | 51 | **2.27** |
| Markdown table | 456 | 315 | **1.45** |

**Decision:** `CHARS_PER_TOKEN = 1.5`, the worst measured case, applied to every
script. Target 350 tokens → 525 chars, hard cap 700 → 1 050 chars.
**Reason:** the hard cap is an invariant the test suite enforces. Planning on prose's
4.1 would let a table or code block reach ~1 900 tokens and breach it — and BGE-M3's own
limit is 8 192. Sizing on the worst case makes the cap structural instead of hopeful.
**Consequence, stated honestly:** prose chunks are smaller than ideal (525 chars rather
than ~1 400). P9 §9.4 tunes chunk size; `docs/eval_report.md` should report whether this
cost recall before anyone raises the constant.
**Rejected:** a per-script table. The density problem is punctuation and identifiers,
not alphabet — a Persian table is as dense as an English one, so a script table would be
the wrong axis.

## D18 — sections are chunked independently and never merged (P3)

**Context:** plan.md P3 allows merging small adjacent sibling sections that fit the
target.
**Decision:** no cross-section merging at all. Each `Section` is chunked on its own.
**Reason:** a chunk's `section_path` is what a citation reports. Merging would make a
citation name two sections, or pick one arbitrarily. The cost is more, smaller chunks
(67 chunks for a 67-section document) — cheaper than an ambiguous citation.
**Rejected:** sibling merging, because the savings are one chunk per short paragraph
while the ambiguity would surface in every user-facing answer.

## D19 — a sentence-free page is cut mechanically (P3)

**Context:** the first implementation split only on sentence boundaries. A
table-of-contents page of `clean-code-excerpt.pdf` contains dot leaders and no
terminators at all, so it stayed one 1 403-char chunk — 60 chunks in that book breached
the hard cap.
**Decision:** two passes. Pack sentences up to the target; then cut any span still over
the hard cap at fixed intervals with the configured overlap. The second pass is what
makes the cap hold for *any* input, not just prose.
**Evidence:** `test_no_chunk_exceeds_the_hard_cap` went from 60 breaches to 0, and the
gate re-run over all five fixtures reports `over_cap=0` for 1 225 chunks.

## D20 — audit pass: dead flexibility removed before P4 (D20)

A repo-wide over-engineering audit ran against P0–P3. Verified, then applied:

- **`INVISIBLE_DROP` deleted.** All 14 hand-enumerated codepoints are Unicode
  category `Cf`; so are ZWNJ and ZWJ. One category test with a two-character keep-set
  replaces the whole table, and the keep-set is now explicit about *why* those two
  survive.
- **`_VERSION_RE` + `_is_version` deleted.** They guarded a case `rstrip(".")` already
  handles: `v2.3.1`, `3.14` keep their dots, `sentence.` and `ERR-404.` lose them.
- **FTS quote-escaping deleted.** `_TOKEN_RE` cannot emit `"`, so escaping it was dead.
- **`EN_STOPWORDS`/`FA_STOPWORDS` collapsed** into one `STOPWORDS`; the split had no reader.
- **Trim hoisted to `parsing/base.trim`**, replacing the same loop in `markdown.py` and
  `text.py`. **`basename`/`stem`/`extension`** replace two near-duplicate filename
  splitters. **`line_to_char` inlined** — it had one caller.
- **PDF text and sections built in one pass.** They were computed twice: once by
  `"\n".join(...)` and again by `_page_sections`' running offsets, which had to
  re-derive the same newline accounting by hand.
- **`Chunk.chunk_hash` and `Chunk.language` lost their defaults.** A default of `""`
  yields a chunk with an unusable hash; the field was reordered ahead of the optional
  location fields because dataclasses forbid a required field after a defaulted one.
  This deviates from plan.md §4's field *order* only — §4 fixed the names, not the order.
- **`@runtime_checkable` removed** from `Embedder` and `VectorStore`; both are checked by
  assignment, which needs no decorator. `DocumentParser` keeps it (one `isinstance` test).
- **`numpy` dropped as a direct dependency.** Imported nowhere; it arrives via
  `chromadb`. It remains in `test_domain_purity.FORBIDDEN` as a *string*, which still
  enforces that the domain layer cannot import it.
- **`make smoke|eval|calibrate|rebuild` removed.** All four invoked `qasystem.cli`,
  which does not exist; each failed on invocation. They return with the CLI in P5/P9.
- **Duplicate PDF tests consolidated.** `test_parsing_pdf.py` kept only what needs a
  synthetic PDF (corrupt, wrong magic bytes, no text layer, empty); fixture-wide
  coverage lives once in `test_parsing_pdf_fixtures.py`.

**Not changed, deliberately:** `CHARS_PER_TOKEN`, `chunk_hash`'s normalization, and
`_enforce_cap`'s trailing coverage append each encode a decision recorded above (D17,
D19) that tests depend on. Shrinking them would undo documented behaviour rather than
remove complexity.

**Two bugs the refactor itself introduced, caught by the suite and fixed:** an inverted
keep/drop test that deleted ZWNJ (the D8 behaviour silently broke), and a test asserting
`ai-engineer.pdf`'s blank page was page 1 when it is page 2.

---

## D21 — the embedding client trusts `index` only when it is a permutation (D21)

P4's only real design decision. The provider returns an `index` field per vector, and
the obvious implementation is `vectors[data[i]["index"]] = data[i]["embedding"]`.

**Measured reason not to.** §2.1 recorded that `Gemini-embedding-001` returns
`index: [0,0,0,0]` for a 4-item request. That implementation would put one vector in
slot 0 three times, drop slot 3, and leave a `KeyError` or a silently mislabelled
result. The symptom would appear as bad retrieval quality, not as a crash, so it would
survive a long time before anyone looked.

**Decision.** `index` is used only when it is a genuine permutation of `0..n-1`. Any
other shape — repeated values, missing field, wrong length — falls back to response
order and logs a warning naming the field. Order-preservation is the `Embedder` port's
contract, so the fallback is the safe answer, not a degraded one.

**Verified by mutation, not just by assertion.** Forcing the permutation check to always
pass makes `test_a_non_permutation_index_falls_back_to_response_order` and
`test_a_missing_index_field_falls_back_to_response_order` fail. Forcing `_retryable` to
accept every 4xx makes the 401 and 400 fail-fast tests fail.

## D22 — the per-item cap is enforced locally, so the provider's context error is a backstop (D22)

§2.1 measured that `Bge-m3` accepts a 40 000-char item and rejects a 45 000-char one with
`maximum context length is 8192 token`. `MAX_CHARS_PER_ITEM=20000` leaves 2× headroom.

The first draft of the live probe tried to reproduce the 8192-token error by sending a
100 000-char item. It could not: `plan_batches` raises first. That is the correct order —
we never spend a request to learn something config already knows. The live test now
asserts the **local** refusal, and the provider's context error remains covered by a
`respx` unit test as the backstop it actually is.

A single oversized item raises `ConfigError`-flavoured `ValueError` rather than being
truncated, because a truncated vector stands for text no citation can point at (I6).

## D23 — `_scrub` became public `scrub` so error messages are redacted too (D23)

`logging_setup._scrub` existed for log records only. The client builds error messages
from the provider's response body, and a provider that echoes an `Authorization` header
would turn our error into an I8 leak. Rather than write a second scrubber, `_scrub` is now
`scrub(secrets, text)` and the client calls it on every provider message before that
message reaches an exception. `test_the_token_never_appears_in_an_error_body` pins it with
a body that literally contains the token; the live probe repeats the check against a real
401.

**Error mapping, one place.** 401/403 → `EmbeddingAuthError` (403's message is parsed for
the `Allowed:` list into `details["available_models"]`, §2.1). 429/5xx/timeouts → retried,
then `EmbeddingUnavailableError`. Any other 4xx → `EmbeddingUnavailableError` immediately,
carrying the provider's text. That last choice reuses an existing error rather than
adding a class for one status code: a 400 from this provider means *we* built a bad
request, and the provider's own words are the useful part of the message.

## D24 — live provider measurements re-taken at P4 (D24)

`RUN_LIVE=1 uv run pytest tests/integration/test_embed_live.py` re-verified §2.1 on
2026-10-01, 6 passed:

| Claim | Live result |
|---|---|
| `GET /v1/models` ids | `Gemini-embedding-001`, `Embedding-3-Small`, `Bge-m3`, `Embedding-3-Large` |
| `Bge-m3` dimension | **1024**, matching §2.1 and §2.2a |
| Vectors arrive L2-normalized | ‖v‖ within 1e-3 of 1.0, so ingest skips normalizing |
| Batch order preserved for 4 distinct texts | 4 distinct vectors returned in order |
| A 20 000-char item | accepted, 1024 dims |
| A 100 000-char item | refused locally, no request sent (D22) |
| A wrong token | 401 → `EmbeddingAuthError`, token absent from the message |

Note the model list comes back in a **different order** each call. Nothing may depend on
its order; the client only does a membership test.

---

## D25 — `with self._db:` was a no-op, so nothing ever rolled back (D25)

The first P5 implementation wrapped every write in `with self._db:`. On a connection
created with `isolation_level=None` (autocommit), that context manager is a **no-op**:
`__exit__` calls `commit()`, finds no transaction open, and does nothing. Each statement
in a multi-statement write committed on its own.

`test_a_failed_write_leaves_no_partial_rows` caught it immediately — the version row and
one chunk row survived the injected failure. But the test only passed *by accident*: I
had already added `fail_after` to force a mid-transaction raise, and the assertion on
`chunk_count` was checking the second chunk's absence rather than the first's presence.

**Fix:** an explicit `_txn()` context manager that spells out `BEGIN` / `COMMIT` /
`ROLLBACK`. Autocommit is the right mode for the connection; the transaction boundary
has to be stated. Mutation-verified: making `_txn` commit on the exception path fails
`test_a_failed_write_leaves_no_partial_rows` and
`test_a_rolled_back_staging_can_be_retried`.

**The lesson worth keeping:** a rollback test that passes for the wrong reason is worse
than no test, because it certifies I3. Mutation testing is what distinguishes the two.

## D26 — the `status` predicate in `eligible_chunks` was untested defence (D26)

The view has two guards, per plan.md §6:

```sql
JOIN documents d ON d.doc_id = c.doc_id
                AND d.status = 'active'
                AND d.current_version = c.doc_version
```

Deleting the `d.status = 'active'` line left **all 54 storage and lexical tests green**.
The reason: `mark_deleted` sets `status = 'deleted'` *and* `current_version = NULL`, and
`NULL = c.doc_version` is never true, so the second guard already excludes the row. The
`status` check was doing no observable work and had no coverage.

It is still worth keeping — it is the guard against a half-applied delete, where
`current_version` survives but the document is deleted. `test_a_deleted_document_is_
ineligible_even_with_a_stale_current_version` now reproduces exactly that state by
setting the column directly, and re-running the mutation fails it. This is the second
time in two phases that a plausible-looking test was passing for the wrong reason.

## D27 — `ensure_collection()` takes no arguments (D27)

The P0 port was `ensure_collection(self, model_id: str, dimension: int)`. `ChromaStore`
is constructed for one `(model_id, dimension)` and derives its collection name from both,
so re-supplying them at call time is an invitation to open the wrong collection with the
wrong vectors — the exact failure I9 exists to prevent.

The port signature is now `ensure_collection(self)`. The isolation guarantee is stronger
than the port ever claimed:

- Different `dimension` → different collection name (`chunks__bge_m3__d1024` vs
  `chunks__bge_m3__d1536`), verified by writing to one and reading zero from the other.
- Different `model_id` → different collection name, same check.
- **Truncation collision** → the one case the name cannot separate. `chunks__` + a
  63-char limit means two long model ids can slugify to the same name, and then the
  collection's stored `model_id` metadata is the only thing standing between a 3072-d
  vector and a 1024-d space. `test_a_name_collision_is_refused_rather_than_mixed` builds
  that collision from two 60-character prefixes and asserts the store refuses. Mutation
  -verified: removing `_assert_isolated` fails it.

## D28 — I10 is tested as "rebuild into an empty index", not "delete the directory" (D28)

Chroma caches per path for the life of the process. Deleting `data/chroma/` from inside
the test process leaves the collection fully populated in memory, so an in-process
`rmtree` test would assert nothing about the guarantee.

What was actually verified, out of band: writing a vector, `rm -rf` the directory, and
opening a **new process** gives `count() == 0`. The operator's flow is real.

The test therefore covers the half the code owns: an index with no vectors is restored
from SQLite's `embedding_cache` with **zero** calls to the embedding API — asserted by
passing a tripwire `embedder` whose `embed` raises. `rebuild` accepts that `embedder`
argument and never calls it, which is the whole point of accepting it. A separate test
rebuilds and then reopens the path to prove the restored index is persistent (L10 + I10
together), and another proves a chunk whose vector is not cached is *skipped* and
reported by `plan_reconcile` rather than half-restored.

## D29 — Chroma metrics confirmed against the live library (D29)

§2.2 was re-measured against the pinned `chromadb==1.5.9` before the adapter was
written, and each observation has a test:

| Observation | Test |
|---|---|
| identical vector → distance `0.0`, orthogonal → `1.0`, so `similarity = 1 - distance` | `test_cosine_distance_is_converted_to_similarity` (identical, orthogonal **and** opposite) |
| `n_results` above `count()` is silently clamped | `test_n_is_clamped_to_the_collection_count` — clamped explicitly anyway, so `n` keeps its meaning |
| querying an empty collection returns empty lists, not an exception | `test_querying_an_empty_collection_returns_nothing` |
| collection `metadata` survives reopen and is readable after a client restart | `test_vectors_survive_a_restart` |
| `heartbeat()` returns an int timestamp | `test_ping_succeeds_on_a_healthy_store` |
| ids containing `:` (`handbook:v3:7`) are legal | `test_ids_keep_their_documented_shape` |
| `upsert` replaces in place rather than duplicating | `test_upsert_replaces_a_vector_in_place` |
| `get(limit=, offset=)` pages correctly | `test_list_ids_is_paged_but_complete` |

**L2 verified across real processes.** `filelock==4.0.7` with `timeout=0` was checked
by running a holder process and a second process against the same directory: the second
raises `StorageLockedError` (`STORAGE_LOCKED`, 503). An earlier version of this test
passed for the wrong reason — the holder held its `DataLock` as a *temporary*, which was
garbage-collected during the sleep and released the lock, so the "second" process
succeeded. `HOLDER` and `THIEF` are now module-level scripts with a named reference, and
`test_the_lock_dies_with_the_process_that_held_it` proves a crashed holder does not leave
a directory permanently unownable.

**Not split, deliberately.** `sqlite_store.py` is 559 lines against the plan's "roughly
400" target. The alternative was a `Database` base class plus a second subclass and a
second connection to the same file — an inheritance layer, an extra connection, and a
third parameter on `rebuild`, all to satisfy a number. The file is one cohesive
repository over one SQLite database with a one-line docstring per method; splitting it
would be reorganizing rather than simplifying. P11 revisits this with the whole codebase
in view.

---

## D30 — live measurements, and a correction to §2.2a's retrieval claim (D30)

`RUN_LIVE=1 uv run python scripts/measure_provider.py` re-measured the provider and,
more importantly, re-measured **retrieval on the committed fixtures** with chunking-
independent phrase gold (plan.md §9.1's design) instead of a document-level label.

**§2.2a's `R@1 = 12/12, MRR = 1.000` is not reproducible and is withdrawn.** On the
committed corpus, 12 answerable questions, gold = an exact phrase that must appear in
the retrieved chunk:

| strategy | R@1 | R@3 | R@5 | MRR@5 |
|---|---|---|---|---|
| dense only | 0.67 | 0.83 | 0.92 | 0.757 |
| lexical only | 0.42 | 0.67 | 0.75 | 0.531 |
| hybrid 0.9/0.1 | 0.67 | **0.92** | 0.92 | **0.764** |
| hybrid 0.8/0.2 | 0.67 | 0.83 | 0.92 | 0.757 |
| hybrid 0.7/0.3 (plan default) | 0.67 | 0.75 | 0.83 | 0.715 |
| hybrid 0.5/0.5 | 0.67 | 0.75 | 0.83 | 0.729 |
| hybrid 0.3/0.7 | 0.67 | 0.67 | 0.75 | 0.688 |

The old figure came from 75 chunks across three small documents with no phrase gold.
The committed corpus is 1 225 chunks and one 393k-char book is **84% of it**, which is a
different measurement. The dimension (1024) and the cross-lingual behaviour are
unaffected; only the retrieval-quality claim changes.

**What the numbers say, and what they do not authorise.**

1. **Hybrid helps only at low lexical weight.** 0.9/0.1 lifts R@3 from 0.83 to 0.92 and
   MRR from 0.757 to 0.764; 0.7/0.3 *loses* R@3 (0.75) and R@5 (0.83) against dense
   alone. Lexical's own R@1 is 0.42, so at 30% weight its noisy top-1 displaces correct
   dense hits. The plan's own framing — "treat 0.7/0.3 as a baseline to evaluate, not a
   truth" — is vindicated, and the baseline looks too high.
2. **The default is *not* being changed on this evidence.** 12 questions means one
   question is worth 0.083 R@1, and the R@3 gap is two questions. Tuning a shipped
   default on that is exactly the unjustified claim §9.3b warns about. `DENSE_WEIGHT`
   stays 0.7 until P9's 50-question dev split can decide it, and this table is the first
   thing that split should reproduce.
3. **Document size skews dense retrieval.** On a corpus capped at 40 chunks per
   document, R@1 is 0.75; on the full corpus it is 0.67. One query moves from rank 6 to
   rank 39 purely because one document grew. This is a direct argument for P7's
   over-fetch and for the evidence gate not trusting dense similarity alone.

**A measurement bug worth recording.** The first version of this comparison fused
*full-corpus* dense ranks against the top-30 lexical hits and reported hybrid as worse
than dense on every metric. plan.md §7.1 fuses the **union** of dense top-60 and lexical
top-30. Fusing 1 225 ranks lets a single rank-1 lexical hit outrank a correct dense hit,
which measures a system nobody is going to build. The table above is the §7.1-faithful
version. The lesson: a measurement of the wrong system is worse than no measurement,
because it looks like a finding.

**Provider facts, re-measured and now tighter than §2.1:**

| Claim | §2.1 | Measured now |
|---|---|---|
| `Bge-m3` dimension | 1024 | 1024 |
| Vectors L2-normalized | ‖v‖ = 1.0000 | min 1.000000, max 1.000000 |
| Single-item ceiling | 40 000 ok, 45 000 fails | **40 949** (binary search, 15 probes) |
| `MAX_CHARS_PER_ITEM=20000` margin | "2x headroom" | **2.05x** — keep |
| Optimal items per request | 32 (assumed) | **32** — measured peak, 29.7 chunks/s vs 23.6 at 64 |
| Sustained ingest rate | not measured | 1 025 chunks / 33 requests, 8 006 chars/s, 31 req/min |

`MAX_ITEMS_PER_BATCH=32`, `MAX_CHARS_PER_ITEM=20000` and `RATE_LIMIT_PER_MIN=100` are all
confirmed by measurement rather than assumption. The 60-second burst cap and the 100/min
limiter leave headroom against a measured 31 req/min of real ingest work.

**The largest inefficiency found is chunk size, not batching.** Cost per character falls
steeply as items get longer. From the item-length sweep, all at 16 items per request so
the batch size is not a confound:

| chars/item | median s | chars/s | ms per 1 000 chars |
|---|---|---|---|
| 200 | 0.666 | 4 805 | 208.1 |
| 500 | 0.840 | 9 524 | 105.0 |
| 1 000 | 1.058 | 15 123 | 66.1 |
| 2 000 | 1.148 | 27 875 | **35.9** |
| 4 000 | 1.357 | 47 163 | 21.2 |
| 8 000 | 1.410 | 90 780 | **11.0** |

So 500 → 2 000 chars is **2.9x** cheaper per character, and 500 → 8 000 is **9.5x**.
`CHARS_PER_TOKEN = 1.5` sizes chunks for the densest measured text (tables), so the real
corpus averages 498 chars against a 40 949-char budget. Raising it trades throughput
against retrieval precision, which is a P9 experiment on the proper dataset, not a change
to make on a 12-question spot-check. The harness for that experiment now exists:
`scripts/measure_provider.py` caches fixture vectors on disk, so a chunk-size or weight
sweep costs no API calls after the first run.

> **Correction.** An earlier version of this entry claimed "~11 ms per 1 000 chars at
> 2 000-char items, roughly 6x cheaper". Both numbers were wrong: 11 ms is the **8 000**-char
> row, and the 2 000-char row is 35.9 ms, i.e. 2.9x rather than 6x. The error came from
> reading the 8 000-char row off the 2 000-char comparison. A 2-item spot-check caught
> nothing here; recomputing from the raw sweep did. The lesson generalises: when a
> measurement is quoted, recompute the derived ratio from the raw table rather than
> carrying a number forward.

## D31 — a fixture the plan called committed was git-ignored (D31)

`plan.md` §2.3 lists `fa/justforfun_book_a4.pdf` as **"Committed. 393 534 chars of
Persian"**, and it is the only proof that NFKC is required. `.gitignore` excluded it.

Consequence, confirmed by cloning the repository and running the suite there: **two
tests failed with `FileNotFoundError`** on any clean checkout, while the working tree —
which always had the file — was green throughout P2, P3 and P5.

```
test_persian_book_is_readable_after_extraction         FileNotFoundError
test_known_limitation_some_lines_lose_their_spaces     FileNotFoundError
```

The file is now tracked. 749 KB is a fair price for the NFKC evidence and for a test that
pins the documented lost-space limitation.

**The lesson is the reason it went unnoticed for four phases.** Every `make check` ran
against the working tree, and the working tree was correct. Nothing was wrong with the
code; the *repository* was inconsistent with its own documentation. `plan.md` P11 lists
"verify a clean clone" as a step, and doing it now, at P5, cost one command. It should be
run as part of `make check` from here on, not saved for the last phase.

**The same fixture is also why the retrieval measurement was hard.** At 1 025 of 1 225
chunks it dominates the index and costs R@1. Committing it makes that visible to anyone
who runs the eval, which is the correct outcome — the imbalance is a property of the
corpus, and hiding the file would have hidden the problem.

---

## D32 — the Markdown parser did not normalize CRLF, which broke I4 (D32)

P6's I4 test — "bytes yielding identical parsed text must return `unchanged` with zero
embedding requests" — had no honest way to be written, because the Markdown parser kept
`\r\n` verbatim in `source_text`. The TXT parser has normalized newlines since P2;
Markdown did not.

Consequence: re-uploading the same document from a Windows editor, or through anything
that rewrites line endings, published a **new version** whose only difference was invisible
whitespace. Every citation, every chunk hash, every embedding cache key changed. That is
exactly the "outdated content" churn I1–I2 exist to prevent, manufactured by a line ending.

Fixed at the right place — one `.replace("\r\n", "\n")` in `MarkdownParser.parse`, before
anything measures a line, so offsets, line numbers, and chunks all agree. `utf-8-sig`
already dropped the BOM.

> Found by a test written for a different reason. The I4 test was supposed to pin
> `parsed_hash`; it could not even be written, and the reason it could not be written was
> the bug. A test that is hard to write is often reporting that the code is wrong.

## D33 — embedding happens outside the write lock, and before any version exists (D33)

The embedded string is `"{title} > {breadcrumb}\n\n{chunk text}"` and contains **no version
number**. So the network-bound part of ingestion does not need the lock, does not need a
version, and cannot be invalidated by a concurrent ingestion of the same document.

Order is therefore: parse → chunk → hash → **embed** → *lock* → re-check unchanged →
allocate → stage → verify → publish → clean up.

Two properties fall out:

- Two documents ingest concurrently without serialising on each other's network calls.
- A failure during embedding costs **nothing**: no version number is allocated, no row is
  written. `test_7b` asserts exactly that, and it is the reason the `ingest_log` row for
  that failure shows `new_version = None`.

The unchanged check is repeated *inside* the lock on purpose. A concurrent ingestion may
have published this exact text while we were embedding; the wasted embedding is our loss,
and returning `unchanged` is the correct answer rather than publishing a duplicate version.

**Mutation-verified:** swapping `parsed_hash` for `content_hash` in that check fails
`test_2b`; the difference between the two is precisely D32.

## D34 — best-effort cleanup needs its own assertions, not just a correctness one (D34)

I2 is satisfied by the view: stale and deleted rows are unreachable *regardless* of whether
cleanup ran. That is the point — but it means **every test that only checks retrievability
passes even if cleanup is deleted outright.**

A mutation replacing `self._purge_superseded(...)` with `pass` left all 36 P6 tests green.
Superseded chunk rows, superseded FTS rows and superseded vectors would accumulate
forever, and nothing would notice.

`test_4` now asserts the reclamation directly: v1's chunk rows gone, its version row gone,
no `:v1:` vector left, and `chunks_removed` recorded in the log. Re-running the mutation
fails it.

> Generalisable, and worth stating once: **when a step is deliberately non-load-bearing,
> the correctness tests cannot be the only ones covering it.** A best-effort step needs a
> test that fails when the step is removed, or "best-effort" quietly becomes "never".

## D35 — a failed version keeps its row; a successful one is fully purged (D35)

plan.md P6 rule 4 says a failure before publish leaves the version `failed` with staged rows
removed. The first implementation did `purge_version`, which deletes the version row too —
so the state was never observable and `version_state()` returned `None`.

Two different purges now exist, and the difference is deliberate:

| Situation | Rows | Version row | Why |
|---|---|---|---|
| Superseded or deleted | chunks + FTS + version | gone | the text must not survive |
| Failed before publish | chunks + FTS | **kept, `state='failed'`** | the version number is spent and the audit trail matters |

`ingest_log` already carries the error code and timings; the retained row carries the
identity of the version number that was consumed, which is what stops anyone wondering
whether v2 was ever used. Mutation-verified: swapping `purge_version_chunks` for
`purge_version` fails `test_9`.

## D36 — `embed_requests` is a delta, and `reconcile` returns the plan it found (D36)

Two small reporting decisions that a test caught by asserting the wrong thing first.

**`embed_requests` is a delta.** The counter lives on `CachingEmbedder` and is cumulative,
so a second ingestion reported the *lifetime* total. A reorder-only edit appeared to cost
one embedding call when it cost none — and the test asserting zero was failing for a reason
that had nothing to do with the code. `IngestResult.embed_requests` is now
`after - before`, so the number means "what this call cost", which is the only useful
reading for a rate-limited provider.

**`reconcile()` returns the plan as found, not as repaired.** Recomputing after the repairs
always looks clean, which would make the return value useless for the thing a caller wants
it for: reporting what drifted. `test_9b` asserts `orphaned == ("ghost:v1:0",)` and
`in_sync is False` on the call that fixed it, then asserts a second `reconcile()` is clean.

**Rule 8 is now directly tested** (`test_9c`): a vector for a document SQLite has never
heard of, and a vector claiming a version SQLite never allocated. Both are deleted from the
index, and neither causes a document row or a version to be invented. A mutation that
inserted a `documents` row per orphan fails `test_9b` and `test_9c`.

## P6 mutation results

| # | Mutation | Test that caught it |
|---|---|---|
| M1 | skip the post-upsert VERIFY step | `test_8b_...silently_loses_vectors...` |
| M2 | unchanged check on `content_hash` not `parsed_hash` | `test_2b_...identical_parsed_text...` |
| M3 | let a cleanup failure become fatal | `test_5_...chroma_cleanup_fails` |
| M4 | drop `current_version = NULL` from delete | `test_5b_...purges_chroma_when_cleanup_succeeds` |
| M5 | fully purge a failed version | `test_9_...left_staging_is_failed_on_reconcile` |
| M6 | skip superseded cleanup entirely | `test_4_...old_text_is_unretrievable` |
| M7 | diff with set instead of multiset semantics | 2 tests in `test_ingest_diff.py` |
| M8 | `reconcile` invents documents from Chroma | `test_9b`, `test_9c` |

M4 and M6 were both **green before** their tests existed. Neither would ever have been
caught by a test written only from the plan's list of required behaviours.

---

# P7 — retrieval, evidence gate, extractive answering

## D37 — `normalize_for_index` deleted newlines instead of spacing them (D37)

The most consequential defect this project has found, and it was four months of prose deep in
P1's code, invisible until P7 tried to cite a Persian PDF.

`Cc` (Unicode "control character") was in the drop set. `Cc` holds the invisible characters the
set was written for — but it also holds `U+000A` newline and `U+0009` tab, which are
**whitespace**, not nothing. The filter deleted them *before* the final
`" ".join(text.split())` could turn them into separators, so the last word of every line was
fused onto the first word of the next:

```python
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(LETTER_FOLD).translate(DIGIT_FOLD)
    text = "".join(ch for ch in text if ch not in _DROP_CATEGORIES)   # ← \n died here
    return " ".join(text.split())                                    # ← too late
```

A Persian bullet list came out as one token:

| in the document | indexed as | matched a query for |
|---|---|---|
| `●پردازش\nاسناد: پشتیبانی` | `پردازشاسناد` | `پردازش` → no. `اسناد` → no. |

Blast radius, all of it silent:

- **Every FTS token for every multi-line chunk was wrong.** Lexical retrieval — including the
  §2 measurement of `lexical R@1 0.42` — was run partly against garbage tokens.
- **`chunk_hash` was not whitespace-insensitive**, contrary to P3's own docstring: `"a\nb"`
  hashed as `ab`, `"a b"` hashed as `a b`, so two textually identical chunks could disagree.
- **Gate coverage was unmeasurable** on exactly the bilingual corpus the gate exists to serve.

Fix, at the one place all callers route through: `Cc` characters become a space, the rest of
the drop set (`Cf`, `Mn`, `Me`) is unchanged, and ZWNJ/ZWJ are still preserved.

```python
        elif unicodedata.category(char) == "Cc":
            kept.append(" ")  # a control character is whitespace, not nothing (D37)
```

Found by a test written for a completely different reason: P7's plan requirement is "a Persian
question cites a Persian source with correct section, page, and line", so the test needed a
Persian PDF to be answerable. It refused, with `token_coverage = 0.00` on **every** chunk —
including the one containing the answer. Coverage of exactly zero is the tell: retrieval had
found the chunk, so the *tokens* were the problem, not the ranking.

Seven regression tests in `tests/unit/test_normalize.py`, because the generalisation is easy to
get wrong again: `\n`, `\r\n`, `\t`, `\x0b`, `\x0c`, `U+2028`, `NUL`, `BEL`, idempotence across
layouts, and `chunk_hash` now ignoring line layout as P3 documented.

**The lesson is the same one as D32, and worth stating once:** both bugs were invisible because
no test's *premise* touched them. A test that is hard to write is often reporting that the code
is wrong — and so is a test that passes while measuring something adjacent to what it claims.

## D38 — ranks are 1-based, everywhere (D38)

`LexicalHit.rank` came out of P5 as `enumerate(rows)` — zero-based. RRF computes
`w / (k + rank)`, so a zero-based rank divides the single best lexical hit in the set by
exactly `k`, giving it the largest boost available. That is precisely the artefact reciprocal
rank fusion exists to remove, and it was one off-by-one away.

Found when `fuse()` rejected `rank=0` and the P7 integration test could not get past setup.

Fixed at the source: `LexicalHit.rank` is 1-based, its docstring says why, and the two P5
assertions that pinned `0` / `[0, 1]` became `1` / `[1, 2]`. One convention for ranks across
the system; a hidden `+ 1` at the single call site would have been the alternative and would
have been found again in six months.

## D39 — fusion is over the candidate union, and weight 0 disables an arm (D39)

Two properties of `retrieval/fusion.py`, both mutation-verified.

**Union, never corpus ranks.** §2 records that fusing full-corpus dense ranks made hybrid look
*worse* than dense (R@3 0.75 vs 0.92) — a "finding" that measured a system nobody would build.
Nothing in P7 could have prevented that regression on its own, because `fuse()` is a pure
function and cannot know whether its caller handed it window ranks or corpus ordinals.

`test_fusion_ranks_within_the_window_never_over_the_corpus` therefore pins the caller's
behaviour on a 40-chunk corpus whose answerable chunk is **last**: its dense rank is 3 and its
lexical rank is 1, its fused score is `0.7/63 + 0.3/61`, and that is explicitly **not**
`0.7/99 + 0.3/99`, which is what corpus ordinals would have produced. Swapping the ranks for
corpus ordinals fails it.

**A weight of 0 switches an arm off.** §9.2 needs dense-only and lexical-only numbers. A
"lexical-only" measurement that still admitted dense-only hits at score 0.0 would not be
lexical-only, and the resulting number would silently mix two systems in a comparison whose
whole purpose is to separate them. `fuse()` now skips a zero-weight arm entirely. I wrote the
test asserting the opposite first — that a zero weight "contributes nothing but still yields
candidates" — which is what a naive reading of "union" suggests, and it was wrong for the one
caller that matters.

## D40 — the gate judges the whole window; `top_k` sizes the answer only (D40)

The first implementation called `retrieve(..., limit=top_k)` and gated the result. So with
`top_k=1` the gate saw exactly one chunk, and a well-covered chunk ranked second could not
rescue a question the system could answer. Truncation is an **answer-size** decision; making it
before the gate let ranking decide the question, which is the gate's job.

`answer()` now retrieves the full `CANDIDATES_N` window, gates it, and passes only
`candidates[:top_k]` to the answerer. `test_the_gate_sees_beyond_the_answer_size` asserts the
gate received more than `top_k` candidates and that coverage 1.0 was among them.

**This one is not reproducible with the fake embedder**, and the test says so. A bag-of-tokens
embedder makes "more token overlap" imply "better dense rank", so a chunk that is ranked low
*because* it covers few of the query's words cannot be constructed. With BGE-M3 it is the
ordinary case: dense R@1 is 0.67 (§2), so a semantically close chunk routinely covers few
query words. The test asserts the observable contract — the gate's input size and the coverage
it received — which is exactly what the mutation `limit=top_k` breaks.

## D41 — gate coverage is the maximum over candidates, not the top candidate's (D41)

`token_coverage` in the verdict is `max` over the window. The tempting alternative is to read it
off the highest-fused candidate, which is one line shorter and wrong: it makes the gate depend
on the very ranking the gate exists to check, so a fusion bug would present as a gate refusal
with no signal that the ranking was at fault. Maximum coverage asks the question the gate
actually means — *does any retrieved chunk contain the query's words* — and does not care which
one the fuser liked best.

`max_dense` and `lexical` follow the same rule, for the same reason.

## D42 — thresholds are keyed by `model_id`, and the failure modes differ on purpose (D42)

`config/thresholds.json` ships conservative placeholders with `"calibrated": false`. A
similarity of 0.6 is a statement about the model that produced it, so `load_thresholds` refuses
a file whose `model_id` is not the embedder's — `ConfigError`, loud, no fallback. That is the
I9 rule one layer up, and P9's `make calibrate` must re-run rather than inherit.

But a **missing or corrupt** file falls back to the uncalibrated defaults with a warning. The
distinction is not inconsistency: "we have no calibrated thresholds" and "we have another
model's real thresholds" are different failures, and only one of them makes a number wrong
rather than merely uncalibrated. Refusing to start over a placeholder would make the system
useless rather than cautious, and the defaults err toward **refusing more**, never less.

`/ready` will report the flag; the README must not claim calibration before P9.

## D43 — what token coverage costs, measured (D43)

Coverage is a fraction of the question's non-stopword tokens present in a chunk, and there is
no stemmer (P1's decision, P9's to revisit). Measured on this corpus:

| question | source wording | coverage |
|---|---|---|
| `kernel version` | `It checks the kernel version first.` | 1.00 |
| `What does the service return when the token is invalid` | `The service returns ERR-404…` | 0.75 |
| `ERR-404 token invalid` | verbatim | 1.00 |
| `سرویس در صورت نامعتبر بودن توکن` | verbatim | 1.00 |
| `سرویس چه خطایی برمی‌گرداند` | `…خطای ERR-404…` | 0.71 |
| `Which Linux distribution does the installer support` | `Run the installer on Linux.` | 0.50 |

Two things this says, both P9's problem rather than P7's:

- **English inflection costs 0.25.** `return` vs `returns`. P7's test thresholds are set from
  these measurements, so the test suite documents the real shape of the signal rather than a
  number chosen to make a test pass.
- **Persian has no stemming, and the gap is Ezafe.** `خطایی` (the question's "error") against
  `خطا` (the source's) scores zero. ZWNJ compounds are fine — D8's split-components behaviour
  makes `می‌رود` match `می رود` — but suffixes are not handled at all. **This is the single
  largest known weakness of the lexical arm, and §9.1's bilingual dataset must include Persian
  questions whose terms are inflected differently from the source**, or the eval will report a
  retrieval quality that is really a morphology mismatch.

Also measured, because the fake embedder's scale matters for every threshold in the test suite:
a two-token question against a 40-character chunk scores cosine **0.19–0.32** with the hashed
bag-of-tokens, where BGE-M3 scores 0.7+. So `min_dense = 0.05` in the P7 tests and `min_coverage`
carries the decision. That is not a loosened threshold — it is the correct acknowledgement that
the fake embedder cannot discriminate, and it is why the shipped `min_dense` is 0.62.

## D44 — `test_3b` passed for the wrong reason (D44)

The I7 test stripped the `[n]` markers out of the rendered answer and compared the remainder to
the concatenated segments. Lossy: the fixture's quoted sentence already ended in a period, so a
mutation that **stripped the period and appended one after the marker** produced a byte-identical
string after stripping. A paraphrase mutation passed a test whose entire claim was "no
paraphrase".

Rewritten to pin the rendering exactly — `answer == " ".join(f"{text} [{id}]")` — on a source
whose sentence has **no** terminal punctuation, so any added full stop is detectable. Re-run
against the mutation, it fails. The lesson repeats D25 and D26: a test that cannot fail is
worse than no test, because it certifies the property.

## What a citation can honestly carry

Measured across the three parsers: `page` comes from PDFs, `lines` from Markdown and TXT, and a
PDF's text layer has no line numbers at all — so `lines` is `null` for a PDF rather than
invented. The section path is real in both cases (`ai-engineer > page 4`, `Handbook > Install`).
`test_6b` asserts the PDF case *including* `lines is None`, so the gap is a pinned fact rather
than a surprise for a caller.

`source_name` was added to the `eligible_chunks` view so that one statement returns the text and
everything a citation quotes about it. That is not a convenience: it removes the window between
deciding a chunk is eligible and reading its text, in which a publish could otherwise slip in.

`has_eligible_chunks()` exists only to separate `empty_knowledge_base` from
`no_relevant_content`, and is read **only** on the path that has nothing to answer with — so the
happy path pays nothing for the distinction.

## P7 mutation results

| # | Mutation | Caught by |
|---|---|---|
| M1 | dense path reads `chunks` directly, no `eligible_chunks` join | 44 tests, incl. `test_5_superseded_text_...`, `test_5b_deleted_text_...` |
| M2 | segment text re-typed instead of sliced from the chunk | `test_6_a_persian_...`, `test_6b_...`, `test_8_hostile_fts...` |
| M3 | gate passes with zero candidates | `test_no_candidates_cannot_pass_...` |
| M4 | newline deleted again instead of spaced | 3 tests in `test_normalize.py` |
| M5 | zero weight only zeroes a contribution | `test_a_zero_weight_arm_switches_off_entirely`, `test_a_single_arm_baseline_...` |
| M6 | fusion over **corpus ordinals** instead of the window (D30) | `test_fusion_ranks_within_the_window_never_over_the_corpus` |
| M7 | `top_k` truncates the gate's input | `test_the_gate_sees_beyond_the_answer_size` |
| M8 | pad the answer with sentences below `min_overlap` | `test_3e_an_unrelated_sentence_is_never_padded_in` |
| M9 | segments no longer in source position | `test_3c_segments_are_ordered_by_source_position` |
| M10 | a citation offset that does not match its excerpt | `test_a_citation_points_at_real_source_offsets` |
| M11 | a refusal carries its closest chunks as citations | 6 tests, incl. all of plan test 2 |
| M12 | thresholds for another model silently reused | `test_thresholds_for_another_model_are_refused` |
| M13 | the rendered answer paraphrases the slice | `test_3b_..._exactly_the_slices_plus_their_markers` (after D44) |
| M14 | segments merged across a gap of real text | `test_3f_sentences_separated_by_real_text_are_not_merged` |
| M15 | an empty question is served the whole corpus | 5 cases of `test_a_question_with_no_terms_is_a_422` |

M6, M13 and M14 were **green before their tests existed**. M13 in particular was green *because*
the test's comparison was lossy, which is the failure mode D44 is about.

---

# Live verification against the real provider

The first time P0–P7 code met real BGE-M3 vectors. **Four defects, all invisible offline**, plus
one shipped threshold that was measured to be dead. Every answerable question below was verified
present in the corpus by tokenizing it before the run — the first attempt used eight invented
questions and four of them were not in the corpus at all, so the gate was right to refuse them and
the run measured nothing.

## D45 — a Latin word glued to Persian script was unmatchable (D45)

The Persian PDFs in this corpus write Persian and Latin with no space: `مدل‌هایembeddingزیر`
("models-embedding-below") and `●مستنداتOpenAPIهمراه`. `\w` matches both scripts, so the
tokenizer produced `هایembeddingزیر` — one token — and **the word `embedding` did not exist in the
index**. No query could ever match it.

Consequences, all silent:

- The headline claim "English question → Persian source" worked only through the dense arm. The
  lexical arm and **the gate's coverage signal** could not see the term at all.
- `مدل embedding` measured `token_coverage = 0.000` on the chunk containing both words.
- After the fix it measures 1.000.

Fixed where every caller routes through, by treating a **script transition as a word boundary** —
letters only, so `ERR-404`, `v2.3.1`, `bge-m3`, `a1b2` and the ZWNJ compounds are untouched:

```python
        elif char != TATWEEL and unicodedata.category(char) not in _DROP_CATEGORIES:
            if kept and _is_script_boundary(kept[-1], char):
                kept.append(" ")
```

`language.py` now imports `_LATIN_RANGES` from `normalize.py` rather than keeping a second copy.

**This is D37's sibling.** Both are "the tokenizer disagreed with the document about where words
end", and both were found by a test whose premise happened to need a real bilingual document. If
one more parser or corpus arrives, the check to write is not a spot-check of tokens — it is
"tokenize a real line and read the tokens".

## D46 — `embed_requests` counted calls, not requests — a 32× error (D46)

The live ingest line read `total_embed_requests=5` for 1225 chunks. Impossible at
`MAX_ITEMS_PER_BATCH=32`.

`CachingEmbedder.requests` counted calls to `embed()`; `EmbeddingClient.embed()` fans that out into
batches, each one an HTTP request. So the log understated every ingest by up to 32×.

That number is not cosmetic. §2 states "ingest cost is dominated by request count, not characters",
§9.2 reports "ingest embed-request count", and the provider is rate-limited at 120/min. A figure
32× too small makes the cost model and the quota analysis wrong.

Fixed by counting where the requests actually happen and putting it in the port, because the metric
is a contract two phases depend on:

```python
class Embedder(Protocol):
    #: Network requests issued so far. ... must count HTTP requests and not calls to `embed()`
    requests: int
```

`EmbeddingClient._post_embeddings` increments it; `CachingEmbedder.requests` delegates. After the
fix the same ingest reads **43** for the whole corpus (39 at 32 items/batch, plus per-document
char-limit splits and the dimension probe) — which matches §2's "a whole fixture corpus ingests in
33 requests" for a smaller corpus, and confirms it. Cache-hit behaviour is unchanged: the caller
wants a delta, and a hit issues nothing.

## D47 — the gate could not authorise the system's own headline capability (D47)

The worst of the four, and the one live-only.

`token_coverage` counts **shared tokens**. For the cross-lingual case §2 is built around — an
English question against a Persian source — coverage is not merely unmet but **structurally zero**.
The rule required coverage on the dense branch, so the dense arm could never authorise anything on
its own. Measured, English question against the Persian-only corpus:

| question | max_dense | lexical | coverage | verdict (before) |
|---|---|---|---|---|
| which programming language and tools are free to use | 0.519 | 0.000 | 0.000 | **refused** |
| what documentation must accompany the code repository | 0.528 | 0.000 | 0.000 | **refused** |
| what are the important parts to test | 0.469 | 0.000 | 0.000 | refused |

Retrieval was right every time — 0.519 and 0.528 sit well above the irrelevant band (0.359–0.451).
The gate refused correctly-retrieved evidence because the one signal that could corroborate it did
not exist in that situation.

The rule now has three disjuncts for three evidentiary situations, not one rule pretending they are
the same:

```text
passed = (max_dense >= min_dense       AND coverage >= min_coverage)   # dense agrees with terms
      OR (max_dense >= min_dense_alone)                                 # uncorroborated: cross-lingual
      OR (lexical   >= min_lexical     AND coverage >= min_coverage_high)  # exact identifiers
```

`min_dense_alone` must exceed `min_dense` — the branch with nothing supporting it cannot be the
easier one — and `load_thresholds` now refuses a file that violates it, exactly as it already
refused `min_coverage_high < min_coverage`.

## D48 — and the answerer then threw the evidence away (D48)

Fixing the gate was not enough, and this is why a live run beats a mutation check: the gate started
passing and the **answerer** refused next, for the same underlying reason.

`min_sentence_overlap = 0.15` scored every sentence 0 — with no shared token, every sentence's
overlap is 0 — and `select_sentences` returned nothing. So the system retrieved, gated, and cited
correctly, and then returned `insufficient_information` anyway.

Fixed with a branch that runs **only when nothing anywhere qualifies**, so the padding suppression
the plan asks for is untouched in the normal case:

```python
    if not picks:
        # Every sentence scored zero, which means the question and the retrieved text share
        # no token at all -- the cross-lingual case. ... (D48)
        return _quote_best_chunk(...)
```

The fallback quotes the best chunk's first `MAX_ANSWER_SENTENCES` sentences as one contiguous
slice, with a full citation. ponytail: those sentences are chosen by **position, not relevance**,
because with no shared token there is nothing to rank them on. The citation still names document,
version, section and page, so a reader who asked in one language and reads the other can find the
passage. Sentence-level cross-lingual ranking needs an embedding per sentence — P9's
`SENTENCE_RERANK`, to be measured before it lands.

Writing it also produced dead code on the first attempt: a per-sentence loop that merged adjacent
spans, which cannot fire, because a *prefix* of a sentence list never has a gap to bridge. Deleted
in favour of one span from the first to the last chosen sentence.

## The shipped `min_dense = 0.62` was measured to be dead

Against 16 live questions:

| | min | max |
|---|---|---|
| answerable `max_dense` | 0.501 | 0.631 |
| unanswerable `max_dense` | 0.359 | 0.451 |

0.62 sits **inside the answerable range**. Six of eight answerable questions scored below it, so the
dense branch could never be the deciding signal — it contributed nothing, and a future edit could
only have made it worse. It was a plausible-looking number with no measurement behind it.

Now `min_dense = 0.47` and `min_dense_alone = 0.50`, both clearing the measured irrelevant band.
After the change: **7/7 genuinely-answerable questions answered, 8/8 unanswerable refused**, plus
the four live gates.

### The honest cost: one false answer, and no threshold removes it

`قفل کردن نسخه` ("locking a version") is **not** in the corpus — `قفل` does not appear in any
document — and the uncorroborated branch answered it at dense 0.527. So the measured tally over
nine truly-unanswerable questions is **one false answer, 11%**.

And it cannot be tuned away. The three cross-lingual questions that must be admitted score 0.519,
0.528 and 0.469; the false answer scores 0.527. No bar on this signal both admits the first three
and rejects the last. **BGE-M3's cosine, over short queries against these documents, does not
separate "right language, wrong topic" from "wrong language, right topic"** — the two cases differ
in something other than similarity.

This is the plan's requirement 3 (relevant retrieval) and requirement 5 (false-answer rate ≤5%)
in genuine tension, and it is a product decision, not a tuning one:

- raise `min_dense_alone` → the false answer goes, cross-lingual goes with it;
- keep it → cross-lingual works, with a false-answer rate that fails §9.3's target;
- gate on something else for uncorroborated hits — a second lexical probe, a language-aware
  similarity, or the answerable-question rate the human is willing to trade.

**Put to the human in plan.md §12.** What is not acceptable is shipping 0.62 and describing it as
tuned.

## Evidence limits, stated plainly

- **16 questions, hand-picked, mostly from one 7-chunk Persian PDF.** Enough to reject a guess
  that sat inside the answerable range; nowhere near enough to call anything calibrated.
  `calibrated` stays `false`, `/ready` reports it, and P9 grid-searches on the dev split.
- `justforfun_book_a4.pdf` is 1025 of 1225 chunks (84%), so the corpus is not balanced. §2's
  0.359–0.451 irrelevant band was measured on a corpus dominated by one Persian book.
- Retrieval **quality** was not measured here — only whether the gate's signals separate. That is
  P9's R@1/R@3/MRR job, and these numbers must not be quoted as retrieval quality.
- Live tests assert *structure and reachability*, never an exact tuned value, so a provider change
  shows up as a failure rather than as a silently different threshold.

## Live-run mutation results

| # | Mutation | Caught by |
|---|---|---|
| M16 | gate refuses the uncorroborated dense branch (the D47 defect) | `test_a_strong_dense_hit_without_coverage_passes_as_dense_only`, `test_dense_only_is_reported_distinctly...` |
| M17 | `min_dense_alone` below `min_dense` | `test_the_shipped_defaults_keep_both_bar_orderings`, `test_..._below_the_corroborated_one_is_refused` |
| M18 | answerer's cross-lingual fallback removed (the D48 defect) | 3 tests in `test_extractive.py` |
| M19 | fallback ignores the sentence budget | `test_the_fallback_respects_the_sentence_budget` |
| M20 | fallback fires even when a sentence matched (padding returns) | `test_a_matching_sentence_is_preferred_over_an_earlier_one` + 2 integration tests |

---

# P8 — API, observability, operations

Five defects, and again **every one of them needed the real wiring**. Four were found by the
API layer's own tests; one was found by writing a bash script.

## D49 — SQLite connections were thread-bound (D49)

`sqlite3.connect` defaults to `check_same_thread=True`, so a connection created in one thread
raises `ProgrammingError` when used from another. Every offline test passed because the store
was built and used in the same thread.

The moment the app existed, the two were different threads — the lifespan builds the graph, a
request handler uses it — and **every single endpoint 500'd**. Not one of the 480 offline tests
had an app with a lifespan.

```python
        # check_same_thread=False because the connection is built in the lifespan and used
        # from request threads. Safe here because of the two guarantees this project already
        # makes: L2's data-directory lock means one *process* owns the file, and the
        # application write lock means one coroutine at a time mutates it. (D49)
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
```

The safety argument is not hand-waving: L2 and the write lock are the reason there is no
concurrency problem to solve, and Python's `sqlite3` is compiled `SERIALIZED`
(`sqlite3.threadsafety == 3`), so the C layer still serialises access.

> **The generalisable lesson, and the second time it has appeared here.** D45–D48 were all
> "the fake embedder is not the real embedder". D49 is "the test never built the thing". A
> test suite that constructs each collaborator and calls each function *in one thread* can be
> green while the composition is impossible. Composition needs its own test, and the cheapest
> honest one is an app factory plus a real request.

## D50 — `filelock`'s `is_locked` is thread-local, so `/ready` 503'd forever (D50)

`DataLock.is_held` asked `FileLock.is_locked`. That counter is **thread-local**: the lock was
acquired in the lifespan's thread and queried from a request's, so `/ready` reported
`lock_held: false` and returned 503 permanently — while the process plainly held the lock.
Under uvicorn both share one loop thread, which is why this would have looked fine in
development and been a coin flip in production.

```python
        # Deliberately *not* asking ``FileLock.is_locked``: that counter is thread-local, so a
        # readiness check running on a request thread saw "not locked" while the process
        # plainly held it, and ``/ready`` returned 503 forever (D50).
        return self._lock is not None
```

`acquire()` succeeding and `release()` clearing the attribute *is* the truth about this
process, which is the only question L2 asks.

## D51 — an uncalibrated placeholder is not another model's measurement (D51)

D42 refused a thresholds file whose `model_id` differed from the embedder's. Wiring the API
surfaced the consequence: **the fake provider could not start at all**, because the only
thresholds file in the repository belongs to BGE-M3 and no fake-model file will ever exist.

The fix keeps D42's guarantee and draws the line where the difference actually is:

| the file is | for another model | decision |
|---|---|---|
| `"calibrated": true` | measured numbers | **`ConfigError`** — real values must never answer for the wrong model |
| `"calibrated": false` | placeholders | warn, use uncalibrated defaults |

D42's promise is "another model's **measurements** never answer here". A placeholder has no
measurements, so substituting it for a placeholder is not a leak. Before this, `demo` mode was
unstartable; now it starts and reports `calibrated: false`, which is the truth.

## D52 — `SQLITE_PATH` and `CHROMA_PATH` were configured and did nothing (D52)

`build_services` derived both store paths from `data_dir`. The two settings existed in
`config.py`, appeared in `.env.example`, and were silently ignored — so pointing them
elsewhere appeared to work and did not.

Found only because a CLI test wanted to make the store fail, set `SQLITE_PATH` to a
directory, and got a healthy report back.

```python
    sqlite_path = Path(data_dir) / "qasystem.db" if data_dir is not None else settings.sqlite_path
    chroma_path = Path(data_dir) / "chroma" if data_dir is not None else settings.chroma_path
```

`data_dir` remains a test seam that redirects both stores at a tmpdir; without it, the
configured paths are authoritative. A setting that is read but not honoured is worse than one
that does not exist, because it is a lie you can act on.

## D53 — a test the harness was silently overriding (D53)

`test_an_unexpected_error_is_500_with_no_stack_trace` needed the response a *server* sends.
Starlette's `ServerErrorMiddleware` builds the 500 and then **re-raises** so the server can
log it, and `TestClient` propagates that. So the first two attempts — `raise_server_exceptions
=False`, then `ASGITransport` — both still raised, and I was about to conclude the handler was
broken.

It was not. `ASGITransport` needs `raise_app_exceptions=False` as well, and the test now uses
it. Worth recording because the symptom looked exactly like a missing exception handler, and
the "fix" a real team reaches for at 3am is to catch the exception in the route — which would
have broken every other error path to work around a test harness.

## The `Embedder` port declared identity as mutable (D53b)

`model_id`, `dimension` and `requests` were declared as plain attributes. Every implementation
exposes them as read-only properties (`CachingEmbedder` delegates), so the port was describing
something none of them were. They are now properties in the protocol, with the D46 note on
`requests` kept.

## What the smoke test caught that the Python tests did not

`scripts/smoke_test.sh` runs against a **real socket**, a real `uvicorn`, and a real
`data/` directory — the only test in the project that does. It found three things:

1. **`curl -f` aborts on the expected 404.** Half the script asserts 404s, and `-f` makes curl
   exit non-zero, which `set -e` treats as a script failure. Now `status_of` branches on the
   status code instead.
2. **`assert version == 2` is wrong.** Versions are monotonic and never reused, so a smoke run
   against a live store reports v7. The assertion is now `+1`, which is the actual property.
3. **The script could not be re-run.** A second run POSTed into a store that already held the
   document and got a 409 — correct behaviour, wrong test. It now removes its own document
   first, because there is deliberately no delete-all endpoint and a smoke test has to
   establish its own precondition.

And two properties verified only there, both of which the plan requires:

* A **second `uvicorn` on the same `data/` refuses to start** — verified by starting one for
  real and reading the log: `StorageLockedError: ... run a single worker (uvicorn --workers 1)`.
  The first instance kept serving.
* The whole add → query → edit → query → delete → query cycle, asserting **content** at each
  transition rather than status codes: the superseded `ERR-404` is unreachable after the edit,
  the new `ERR-503` is answerable, a re-upload spends zero embedding requests, and after the
  delete both are unreachable.

## A lock holder must keep a reference (re-learned)

P5 recorded that a temporary holding a `DataLock` is garbage-collected and releases it. I
reintroduced exactly that in the new L2 test — `DataLock(path).acquire()` as a bare expression
— and spent a while bisecting whether `filelock` works across processes at all before finding
it. It works fine; the object was simply gone, and with it the file handle and the `flock`.

The new `LOCK_HOLDER` keeps the variable and says why, so the next reader does not have to
rediscover it. Bisecting it was still worth the time: "the lock does not work" and "my test
does not hold the lock" look identical from the outside.

## `httpx` logs full URLs at INFO

`httpx` emits `HTTP Request: GET http://host/path?query=...` at INFO. That is why the
middleware's logging test is scoped to `qasystem.api.app` records rather than asserting over
all captured output. It is not a leak here: the only `httpx` client in the system calls the
embedding provider on a fixed path with the token in a **header**, so no user data can reach
one of its log lines. Noted so the scoping looks deliberate rather than convenient.

## P8 mutation results

| # | Mutation | Caught by |
|---|---|---|
| M22 | `POST /documents` silently replaces instead of 409 | `test_post_different_content_to_an_existing_doc_is_409_and_points_at_put` |
| M23 | `PUT` resurrects a deleted document | `test_put_a_deleted_document_is_404_rather_than_resurrecting_it` |
| M24 | the 500 handler returns the exception message | `test_an_unexpected_error_is_500_with_no_stack_trace` |
| M25 | the middleware logs `str(url)` instead of `url.path` | `test_the_middleware_logs_the_path_and_never_the_query_string` |
| M26 | `/ready` reports a hard-coded model id | `test_ready_reports_every_dependency` |
| M27 | the model guard accepts another model's calibrated thresholds | 2 tests in `test_gate.py` |

M25 was **green before its test existed**. Nothing else in the suite would have noticed
`url.path` becoming `str(url)`, which is a one-character change that quietly starts logging
every user's question.

---

# Whole-project live verification (real BGE-M3, full fixture corpus)

`scripts/smoke_test.sh` had only ever run against the fake provider, the CLI had never touched
the real one, I10's "zero API calls" had only ever been asserted against an embedder that makes a
request for anything, and no live test had ever ingested more than one 7-chunk PDF. Closed all
of that in `tests/integration/test_system_live.py`.

**Result: no new defects.** 103 live tests green, 545 offline green, the full HTTP cycle green
against real vectors. That is worth saying plainly, because it is not what the previous two live
rounds found — and the reason is that the gaps above were *verification* gaps, not code gaps.
What did surface were two **measurements** that P9 needs and that no offline test could produce.

The two failures during this round were both my own test questions, not the system:

| what I wrote | what happened | whose fault |
|---|---|---|
| "what does the story say about the installer checkpoints" | gate refused it | mine — `installer` and `checkpoints` appear in **no** document in the corpus. Verified by tokenizing all five before asking. |
| `delete("justforfun_book_a4")` | `DocumentNotFoundError` | mine — the doc_id is the slug `justforfun-book-a4`, not the file stem |

That is the second time this corpus has caught me inventing a question. The rule that falls out
of it is now in the test file: **verify the terms are in the corpus before asserting the gate
answers.** A refusal on an unanswerable question is the gate working, and a test that treats it
as a failure is measuring nothing.

## D54 — query latency is the embedding API, not the retrieval (D54)

Measured on 1225 chunks with real BGE-M3, decomposed per stage:

| question | embed | chroma | fts | sqlite + fusion + gate + answer |
|---|---|---|---|---|
| `what did the grandmother leave behind` | **1230ms** | 5ms | 0ms | 13ms |
| `what is the boiling point of mercury…` | 286ms | 5ms | 0ms | 21ms |
| `who is Eleanor` | 448ms | 4ms | 0ms | 13ms |
| `BGE-M3` | 379ms | 5ms | 0ms | 16ms |

**Everything this project builds costs 13–21ms. One network call costs 286–1230ms.** Retrieval,
fusion, the evidence gate and answer assembly are together about 2–5% of query latency, and
`embed_calls == 1` on every query, so the 1230ms outlier is provider variance and not a retry.

End-to-end over the full corpus: **P50 ≈ 410ms, P95 ≈ 1.8s**, and the tail is the provider's.

Two consequences for P9, and one warning:

* Optimising retrieval cannot move query latency. `CANDIDATES_N`, `OVERFETCH`, `RRF_K` and the
  gate's cost are all rounding errors against one embedding call. The levers that matter are
  batching and serving the embedder — and, per §2, making `TOP_K` small enough that fewer
  chunks need embedding-based reranking.
* Do not read the P95 as our tail. A load test that reports "P95 1.8s" is reporting the
  provider, and presenting it as ours would be a small lie with a large performance budget
  attached.
* `test_query_latency_is_the_embedding_api_not_the_retrieval` pins the local share under 150ms,
  so a genuine regression in *our* work fails even though the provider still dominates.

## D55 — the cross-lingual miss is the lexical arm, and no threshold can fix it (D55)

The full-corpus measurement: 8/8 unanswerable refused (0% false answers), 7/10 answerable cited
the right document. Two of the three misses are the cross-lingual ones, and both cite the wrong
document while still *citing real text* — which is worse than a refusal, because a caller who
asked in English cannot check Persian text they do not read.

The cause is precise, and it is not the gate:

```
'what language and tools are free to use'
  clean-code-excerpt.pdf   sim=0.471  cov=0.25  score=0.01335   <- rank 1
  justforfun_book_a4.pdf   sim=0.541  cov=0.00  score=0.01148
  ai-engineer.pdf          sim=0.539  cov=0.00  score=0.01129   <- rank 5, the RIGHT one
```

`0.01129 = 0.7/62`: the correct Persian chunk is **dense rank 2 and receives no lexical
contribution at all**, because it shares no token with an English question and so can never
collect a lexical rank. The English chunks collect lexical ranks 1–19 on incidental words
("language", "tools", "use") and overtake it on the sum.

So the dense arm — the thing that makes cross-lingual retrieval possible at all — finds the
right chunk and then loses, because RRF sums ranks and a chunk that is mediocre on both arms
beats a chunk that is *best* on one and *absent* from the other.

**And no dense-similarity bar separates them.** Measured: the wrong chunk sits at **0.547**, the
right one at **0.539**. Raising `min_dense_alone` kills both; lowering it admits more wrong
ones. The parameter D47 introduced cannot solve a ranking problem.

This is the third independent measurement pointing at the same place — §2's finding that 0.7/0.3
beat 0.9/0.1, D30's finding that fusion can penalise the correct answer, and now this. The
honest conclusion is that **for a query whose language differs from the source's, the lexical arm
contributes noise.** The options are §9.4 experiment 1 (weights), or something the weights cannot
express: detect that coverage is zero and drop the lexical arm's contribution for that query.

That is P9's decision, not P7's, so **nothing was changed**. What was done instead is
`test_the_cross_lingual_miss_is_the_lexical_arm_not_the_dense_bar`, which pins today's behaviour
with the explanation attached and fails if the behaviour changes *without* the explanation being
re-measured. It asserts all three of: the Persian chunk is retrieved; it has `similarity > 0.5`
and `lexical_score is None`; and it still loses to an English chunk. If a future change makes it
win, that test fails and demands the reasoning be redone.

## What the corpus-level live run confirmed

* **Batching holds at scale.** 1225 chunks ingest in **43** provider requests: 39 at
  `MAX_ITEMS_PER_BATCH=32`, plus per-document character-limit splits and the dimension probe.
  §2's "a whole corpus ingests in 33 requests" is consistent for a smaller corpus.
* **Every published chunk has a vector**, and `eligible_chunk_ids()` matches — 1225/1225.
* **I2 at scale**: deleting the Persian book (84% of the index, 1025 chunks) leaves the other
  documents answerable. `پردازش اسناد` still resolves to `ai-engineer.pdf` afterwards.
* **I10 measured, not asserted**: `rebuild` restored every real BGE-M3 vector into a fresh
  collection while the real client's request counter did not move. Against the fake embedder
  this test was vacuous; it now is not.
* **The CLI works against the real provider**: `check-storage` round-trips both stores and reports
  `Bge-m3` / 1024; `reconcile` reports in-sync on a real corpus; `rebuild` restores real vectors.
* **P6's cost invariants hold on real vectors**: a one-paragraph edit spent exactly 1 request; a
  section reorder and an identical re-upload spent **0**.

## D56 — a test that pinned exact ranks from an approximate index (D56)

The clean-clone verification caught a real flake: `test_fusion_ranks_within_the_window_never_
over_the_corpus` passed in my working tree and failed roughly **one run in five** in a fresh
clone, with no difference in the code.

The cause was over-specification in the test, not a defect in the system. It asserted
`dense_rank == 3` and `lexical_rank == 1` — exact positions from Chroma's **HNSW index**, which
is an approximate nearest-neighbour search whose tie-breaking is not stable between runs.

The claim that test exists to protect is D30's: *ranks are positions in the retrieved window, not
in the corpus*. That claim does not need those particular numbers. It now asserts:

* both ranks are single digits, where the corpus ordinal is ~40;
* the score is exactly `w_d/(k+rank_d) + w_l/(k+rank_l)` **applied to the observed ranks**;
* that differs materially from the corpus-ordinal score.

Strictly more informative than the constants — it pins the rank's *meaning* rather than its
value — and stable across runs.

This is the third time the same lesson has appeared (D44's marker-stripping comparison, the
middleware logging test, and now this): **assert the claim, not the incidental value.** A
hard-coded constant in a test is a claim that the constant will never change, which is almost
always a stronger claim than the one being made.

> It is also the argument for §0.11's clean-clone gate. The failure was order-and-run dependent,
> so it would have been attributed to flakiness and retried away rather than diagnosed. Running
> the verification in a *fresh* environment is what made it visible.

---

# P9 — evaluation and gate calibration

`tests/eval/` is the harness, `tests/eval/dataset.jsonl` the 50 questions, `tests/eval/corpus/`
the six documents, `tests/eval/report.md` the deliverable and `config/thresholds.json` the
calibration. `make eval` and `make calibrate` run it against the real provider into
`data/eval/`, which is a **separate index on purpose**: the fixture corpus is 84% one Persian
book and would have measured its skew instead of the system.

## The headline

| | dev (30 questions) | held out (20) |
|---|---|---|
| answered on answerable | **0.91** (23 answerable) | **0.93** (15) |
| **false answers** | **0** of 7 unanswerable | **0** of 5 |
| R@1 / R@3 / R@5 | 0.83 / 0.96 / 0.96 | 0.67 / 0.93 / 0.93 |
| gold quoted in the cited text | 0.91 | 0.88 |

Shipped thresholds: `min_dense=0.52  min_dense_alone=0.66  min_coverage=0.40
min_lexical=0.80  min_coverage_high=0.80`. 26 040 feasible operating points of 153 153 tested,
a point being feasible when it refuses every unanswerable dev case. §9.3's target was ≤5% false
answers preferring 0%, and the answer is **0% on both splits** — which reproduces nothing of
D47's *1 in 9*, because that measurement was on a 7-chunk PDF and this is a 49-chunk corpus
with a known answer for every question.

What P9 also found, and it matters more than the table: **a false-answer rate of 0.00 does not
mean the answers are right.** See D60.

## D57 — a grid search that reported "no feasible point" because of its own bounds

The first live `calibrate` run said no operating point could hold a 0% false-answer rate. That
was **false**, and the cause was the search space, not the data: `COVERAGE_GRID` stopped at
0.40 while the discriminating value is ~0.44. Widening it found 26 040 feasible points
immediately.

A grid search's negative result is only evidence about the region searched. The grid now spans
0.30–0.70 on the dense bars and 0.00–1.00 on the coverage bars, and `choose()` raises rather
than returning a best-effort point, so "nothing is feasible" is always a statement about data
and never about a missed bound. The same lesson bit §9.4 experiment 2 (D64).

## D58 — `min_lexical` is a threshold on a saturated signal and cannot discriminate

`LexicalHit.lexical_score` is `bm25 / best_bm25_of_the_same_query`, so the **top hit is exactly
1.0 for every query FTS matched at all**. The gate reads `max(lexical_score)` over candidates,
which is therefore 1.0 whenever FTS returned anything. Measured across all 50 eval questions:
`lexical` is 1.00 for 49 of 50 — every near-topic unanswerable *and* every answerable.

So `min_lexical` can only be 0.0 or 1.0, and its branch (`lexical AND coverage`) is decided by
`min_coverage_high` alone. The parameter is documented as evidence of an exact-term match and
measures nothing. `LEXICAL_GRID` now carries three values instead of seven, because searching a
dead axis harder only looks like rigour.

Not fixed here. The fix is a change to the gate's contract and its debug payload, it would
invalidate P7's live tests, and it buys nothing measurable — the calibrated branch works without
it. Recorded for P10/P11, where the plan already schedules a refactor pass. The shipped
`min_lexical: 0.80` is what the grid chose *given the signal saturates at 1.0*, and
`config/thresholds.json` says so in its notes rather than letting the number look meaningful.

## D59 — fusion weights: 0.9/0.1 measured better than 0.7/0.3, and the lexical arm still earns its 0.1

§2 measured 0.9/0.1 ahead of the 0.7/0.3 default on 12 questions and called it "suggestive, not
decisive". P9 re-ran the sweep on the eval dataset, dev and held-out separately:

| weights | dev R@1 | dev R@3 | dev MRR@5 | held-out R@1 | held-out R@3 |
|---|---|---|---|---|---|
| 0.7/0.3 (was shipped) | 0.78 | 0.91 | 0.848 | 0.67 | 0.87 |
| **0.9/0.1 (adopted)** | **0.83** | **0.96** | **0.884** | **0.67** | **0.93** |
| 1.0/0.0 (dense only) | 0.78 | 0.96 | 0.862 | 0.53 | 0.93 |

Better on dev on all four metrics, worse on none of the held-out ones, which is exactly §9.4's
adoption criterion. Adopted in `config.py` and `.env.example`.

**The lexical arm still earns its 0.1.** Dense-only is worse than hybrid on dev R@1 (0.78 vs
0.83) and much worse on held-out R@1 (0.53 vs 0.67). §2's other finding is now decisive rather
than suggestive: hybrid beats both single arms.

**A property this exposed, verified rather than assumed:** the gate's verdict is *completely
weight-independent* as long as both arms are non-zero. Measured: at all eleven weight points,
the number of cases whose gate verdict differs from the shipped configuration is **0**. The
reason is structural — `max_dense`, `lexical` and `token_coverage` are maxima over the candidate
set, `overlap` is a set intersection of the top-k sets, and the union of the two arms' windows
does not depend on the weights. So changing the weights changes which evidence is quoted, never
whether the system answers. That is why §9.2 could measure single-arm baselines at all (D39)
and why the calibration is valid for every hybrid weight setting.

## D60 — a false-answer rate of 0.00 hid an entire failure mode

`x01` asks *"the exact path of the file that holds the gateway's configuration"*. The path is
`/etc/aurora/gateway.toml`, in the **Persian** deployment guide. The system answered it, with
citations, citing `handbook.md` — the English handbook, which mentions a configuration file and
contains no path. A caller who cannot read the cited language has no way to notice.

Nothing in §9.2's metric list saw it. R@k is about the *candidate window*; the false-answer rate
counts *unanswerable* questions; `answered` counts gate passes. An answered question citing a
document that does not contain the answer satisfied all three.

Added **`gold quoted`**: the share of gold phrases appearing in a chunk the answer actually
cites. 0.91 dev, 0.88 held-out — so **12% of the facts the system claims to have answered, it did
not quote.** The report names every case and classifies the cause, because the three causes need
different fixes:

| case | cause | measurement |
|---|---|---|
| `x01`, `x08`, `m05` | **retrieval** | the citation names a document that does not hold the answer |
| `fa02` | **truncation** | the gold chunk is in the window at position **10**, below `top_k=5` |
| `m01` | **answerer** | the gold chunk is inside `top_k`, but the sentence scores **0.067** against a 0.15 bar while the sentence *beside* it — which talks about the port without containing the number — scores **0.153** and is quoted instead |

`m01` is a limitation of the signal rather than a tuning miss: word overlap cannot connect
"what port" to a literal. Lowering `min_sentence_overlap` to reach it would pad every answer
with unrelated sentences, which §7.3 forbids.

## D61 — cross-lingual retrieval here is luck, not capability

§2 claims English-question-against-Persian-source as the headline property. P9 measured it, and
the same two questions were asked against three indexes:

| index | chunks | `max_dense` | `token_coverage` | answered? |
|---|---|---|---|---|
| `ai-engineer.pdf` alone | 7 | 0.519 / 0.513 | **0.00** / 0.00 | **no** |
| eval corpus | 49 | 0.519 / 0.513 | 0.00 / 0.25 | no / yes |
| fixtures | 1225 | 0.533 / 0.547 | **0.40** / 0.50 | **yes** |

Two findings, and the first refuted the hypothesis I expected:

* **Absolute cosine does not drift with corpus size.** Answerable `max_dense` is 0.513–0.547
  across 7, 49 and 1225 chunks — flat. So thresholds are not a corpus-size problem, and the
  instinct to normalise similarity by corpus size would have been a fix for a defect that does
  not exist.
* **What varies is `token_coverage`, and that is the whole story.** On the 7-chunk index the
  cross-lingual question shares no token with anything, so coverage is structurally 0 (D47) and
  the only branch that can admit it is the uncorroborated `dense_only` one, which needs
  `max_dense ≥ min_dense_alone = 0.66`. It scores 0.519, so it is refused. On the larger indexes
  an **unrelated** document lends the question incidental lexical coverage (0.40, 0.50) and the
  corroborated branch admits it.

So cross-lingual retrieval works on this system when some other document happens to share words
with the question. That is not the property §2 claims, and no threshold on cosine can supply it:
D47's branch must be either reachable (and then it admits `n03` at 0.609) or safe (and then it
refuses the real hit at 0.519). **This is plan.md §12 question 3, now measured on both sides.**

## D62 — a "safer" tie-break turned cross-lingual retrieval off, and the fix is a medoid

When many points tie on dev recall at 0 false answers, `choose()` was picking the *strictest*
value on every axis. That sounds conservative and was the opposite: `min_dense` only gates the
branch that **also** requires `token_coverage`, so a corroborated hit is already vouched for.
Raising it refuses nothing extra — it buys no measured safety and costs transferability. With
`min_dense = 0.60` the fixture corpus's cross-lingual question (0.533) was **refused**; at 0.52,
which the grid showed gives the *identical* 0.913 recall at 0 false answers, it is answered.

Preferring looser then over-corrected and picked `min_coverage_high = 1.0`, an extreme that
silenced the exact-term branch and cost a held-out answer.

Both were the same mistake: treating an extreme on one axis as if that axis were the important
one, when the data says none of them is. The tie-break is now a **medoid** — among points with
equal measured performance and equal safety, the one closest to the centre of the feasible set.
Scale-free, no per-axis argument, and it lands on a point that is ordinary in every dimension at
once, which is what a threshold set wants, because thresholds are applied to corpora nobody has
measured yet.

An earlier version also omitted `min_coverage` from the sort key entirely, so the calibration's
output depended on iteration order: a number that changes when nothing about the data changed is
not a measurement.

## D63 — the harness nearly published a latency number 30× too small

The first two live runs reported **P50 11 ms** "cold". They were not cold: a second
`make eval` finds every question already in `embedding_cache`, so the "first touch of every
question" was nothing of the kind. The real figure is **P50 354 ms, P95 448 ms**, and the
warm-cache figure (P50 10 ms) is the system's own share.

The mistake was in the harness — the thing whose job is to not misattribute latency, committing
the misattribution D54 warns about. Latency is now measured on a throwaway index built exactly
like the real one, on every run, at a cost of ~56 provider requests. Re-running an evaluation
must not change what it measures, and the cheapest way to guarantee that is to never reuse the
state the measurement depends on.

## D64 — §9.4 experiment 2 cannot discriminate on this corpus, and the reason is measured

Chunk size is §2's largest *cost* lever (9.5× cheaper per character at 8 000 than at 500). The
retrieval side is unmeasurable here: the eval corpus is **49 sections in 49 chunks and its
longest section is 635 characters**, below the smallest hard cap tried (1 050). Section
boundaries are atomic and a section is at least one chunk, so the chunk count is fixed by the
corpus at every size and all three rows are 49 by construction.

A grid stopping at 1 000 tokens would have reported "chunk size does not matter" — a conclusion
drawn from a search space that could not have found one, the same failure as D57. The grid now
runs to 4 000/8 000 tokens and the report says plainly that no change is adopted **because
nothing was measured**, and that making the experiment real needs a corpus with a section longer
than the hard cap — a corpus change, not something to smuggle in by inflating a grid.

## D65 — what P9 did not do

* **`SENTENCE_RERANK` was not measured**, because it has no implementation. There is nothing to
  measure, and building a reranker before knowing whether sentence selection is a problem is the
  speculative work §10 rule 16 forbids. D60 shows it *is* a problem for one case in fifty, which
  is a P10 candidate, not a reason to build the feature now.
* **The per-item cap was not measured**, because it cannot affect retrieval quality: it is a
  provider guard, the chunker produces chunks an order of magnitude below the measured 40 949-char
  ceiling, and lowering it would only convert a request that succeeds into one that fails. An
  argument, not an experiment.
* **`min_lexical` was not removed** (D58). It is a change to the gate's contract and debug
  payload, it would invalidate P7's live tests, and it buys nothing measurable. P10/P11.
* **The gate was not redesigned** to add a second signal to the uncorroborated branch. §12
  question 3 reserves that for a human, and the measurement is now in front of them.

## D66 — the human's answers to §12 Q3 and the evidence standard, and what they commit us to

Two questions P9 could measure but not decide. Both are now answered, and both change work
rather than just documentation.

### Q3 (the cross-lingual branch): **give it a second signal**

P9's measurement closed off the alternatives rather than choosing between them. Holding a 0%
false-answer rate *is* giving up the uncorroborated branch: `min_dense_alone` must exceed
0.609 to refuse `n03`, and the real cross-lingual hits measure 0.513–0.547. So (a) and (b) were
never alternatives — the choice was between the calibrated state, (c), and dropping cross-lingual
entirely. **(c) was chosen**, and it is a P10 item with `docs/eval_report.md` as its acceptance
test.

What P9 already eliminated, so P10 does not re-measure it: a second **lexical** probe is the only
viable direction, and an **IDF-weighted coverage** is not. Measured margin over a single threshold
across all 50 questions — plain coverage **+0.86**, `max_dense` **+0.79**, `rare_terms` **+0.45**,
`bm25span` **+0.17**, IDF-weighted coverage **+0.23**. IDF fails because in a six-document corpus
a near-topic question's terms are *all* rare, so weighting them by rarity hands an irrelevant
chunk the same score a relevant one gets.

So the candidates that remain, in the order worth trying:

1. **A cross-lingual lexical probe.** Take the question's tokens, transliterate or
   otherwise project them into the candidate's script, and re-measure coverage. The signal
   `token_coverage` cannot provide is *does the candidate share meaning-bearing tokens once the
   script barrier is removed* — which is exactly what it currently reports as a structural 0.
2. **Script-aware segment matching.** Split each candidate's ZWNJ compounds and compare
   stems, since D43 measured Persian coverage losing 0.29 to Ezafe suffixes. Cheaper than (1)
   and stackable with it.

Either way the acceptance condition is the same and it is the one P9 could not meet: the branch
must answer the four cross-lingual eval cases *and* refuse all twelve unanswerable ones. Until
that holds on both splits, the branch stays as calibrated and cross-lingual stays documented as
unreliable (D61).

### The evidence standard: **hold the system to `gold quoted`**

`false answers` and `R@k` are the plan's stated criteria and both are strong (0 of 12, R@1 0.76).
But D60 showed they are blind to a failure that matters more to a caller: an answered question
citing a document that does not contain the answer. A citation is a claim that its source
supports the answer, so the number that measures requirement 4 of `jobTask.md` is the share of
gold facts present in the text the system quotes — currently **0.91 dev, 0.88 held out**.

`gold quoted` is therefore promoted from an added diagnostic to **the acceptance metric for
requirement 4**, and the three causes D60 named are P10's work list, each separately actionable:

| cause | cases | what it needs |
|---|---|---|
| retrieval — wrong document cited | `x01`, `x08`, `m05` | the §12 Q3 fix, or the lexical-arm correction from D55 |
| truncation — gold chunk at window position 10, below `top_k=5` | `fa02` | a `TOP_K` decision, traded against answer length |
| answerer — sentence scores 0.067 against a 0.15 bar while the sentence beside it scores 0.153 | `m01` | a signal that connects "what port" to a literal; **not** a lower bar, which would pad every answer |

No target number was set, deliberately: 0.88 is the measured floor and the target is to raise it
by removing causes, not by picking a threshold that happens to be passable today. The number is
reported per split, in the report's tables, and in the `eval` command's JSON, so it is visible
without opening the report.

# P10 — hardening

Four items in `plan.md` §9's P10. Coverage on the six target directories was already **90%+**
(the lowest is `answering/sentences.py` at 90%), so the work was the three named fixes and the
two test asks. Two landed, one is a measurement, and one is blocked with the evidence that
blocks it.

## D67 — `min_lexical` removed, and the deletion is provably behaviour-preserving

`plan.md` P10 named this and P9 deliberately deferred it, because it changes the gate's contract.
The change is a deletion, so the contract got smaller:

```diff
- passed = ... OR (lexical >= min_lexical AND coverage >= min_coverage_high)
+ passed = ... OR (coverage >= min_coverage_high)
```

`lexical_score` is `bm25 / best_bm25 of the same query`, so the top lexical hit is exactly 1.0 for
every query FTS matched at all — measured at **1.00 for 49 of the 50 eval questions**, answerable
and unanswerable alike. The parameter had one live setting and its branch was decided by
`min_coverage_high` regardless.

The clause could only ever have refused in one state — `coverage > 0` with no lexical hit — and
**that state is unreachable**: `lexical_score` is the max over the *lexical* hits, so it is 0.0
only when FTS returned nothing, which happens only when the question has no searchable terms, and
`token_coverage` is then 0.0 by definition.

**The unit test that pinned this asserted a fiction for a year.** It hand-built
`coverage=1.0, lexical=0.0` and expected a refusal, which is precisely the unreachable state. The
invariant is now checked against a real index and real questions in
`test_the_unreachable_state_is_unreachable` — a unit test can only assert it by constructing the
signals, and constructing them is what produced the fiction. Caught while doing the deletion.

Two confirmations that the axis was dead: the calibration grid shrank from 153 153 tested points
to 76 100, and the **medoid operating point is unchanged** (0.52/0.66/0.40/0.80) with identical dev
and held-out numbers.

`config/thresholds.json` drops the key and moves to version 3. A copy still carrying the retired
key loads unchanged, since only the named fields are read. The version is now derived from the
file on disk: it was `thresholds.version + 1` off a grid point built at `version=1`, so every run
wrote `2` forever and a changed rule would have shipped under the old rule's number.

## D68 — `TOP_K` is the wrong knob, and D60's own cause labels were incomplete

D60 named three causes for the cases where the system answers without quoting the fact. One was
"truncation": `fa02`'s gold chunk sits at window position 10 with `top_k=5`, so raising `TOP_K`
was the obvious fix. Measured, it is not:

| `top_k` | gold quoted | answer chars | false answers | causes |
|---|---|---|---|---|
| **5** (shipped) | 0.897 | 240 | 0 | truncation 1, retrieval 2, answerer 1 |
| 10 | 0.897 | 271 | 0 | answerer 2, retrieval 2 |
| 15 | 0.923 | 286 | 0 | answerer 1, retrieval 2 |
| 20 | 0.923 | 295 | 0 | answerer 1, retrieval 2 |

The truncation cause **converts** to an answerer cause rather than being fixed, and the whole
sweep buys one gold fact out of 43 for +19% answer length. `TOP_K` stays 5.

Digging into why produced a finding that corrects P9. `fa02`'s gold sentence is a Markdown code
fence:

```text
DROP overlap=0.000  '```\nauroractl install --channel stable\n```'
```

The question is *"which command installs the gateway?"* — in words — and the answer is a command,
sharing no word with it. The word that connects them (`نصب`) is in the section heading, not the
sentence. So `fa02` has **two independent blockers** and removing either one fixes nothing.

**D60's three causes are therefore two.** Every non-retrieval case is the same defect: word
overlap cannot connect a question asked in words to an answer given as a literal. `m01` (0.067)
and `fa02` (0.000) are that failure at different distances. It is a limitation of the signal, not
a tuning miss — §7.3 forbids padding answers with unrelated sentences, so the bar stays where it
is, and §9.4's `SENTENCE_RERANK` was reserved for exactly this and had no implementation to
measure.

The classifier now prints the answerer's **own** overlap score for the sentence holding the fact,
computed with the answerer's formula. The label was not wrong about the mechanism at `top_k=5`; it
was incomplete, and only a number says so. It also dropped one case (`x08` is now refused, so it
is not an answered answerable), leaving four.

## D69 — the §12 Q3 second signal is not achievable on this dataset, and the reason is not the gate

`plan.md` P10 set the acceptance condition: **answer all four cross-lingual eval cases and refuse
all twelve unanswerable ones**, on both splits. Four signals were measured before concluding,
because the cost of guessing wrong here is a shipped rule fitted to four data points.

| signal | min over cross-lingual | max over unanswerable | separates? |
|---|---|---|---|
| `max_dense` (cosine) | 0.482 | 0.609 | no |
| "is the top hit another language?" | 2 of 4 produce it | 2 of 12 also produce it | no |
| cosine restricted to other-language candidates | 0.482 | **0.541** | no |
| bm25 restricted to other-language candidates | 0.000 | 24.53 | no |

**The language-aware signal that §12.3 offered costs nothing and still fails.** `chunks.language`
is already stored, so "the best hit is in a different language from the question" is three lines.
It is 2/12 on the unanswerable side — an off-topic Persian question's best English neighbour is
often genuinely in another script, which is the whole point of a bilingual corpus.

**And IDF-weighted coverage, which P9 had already ruled out, was the right instinct anyway**: in a
six-document corpus a near-topic question's terms are *all* rare, so weighting them by rarity hands
an irrelevant chunk the same score a relevant one gets. P9 measured it at a +0.23 separation margin
against plain coverage's +0.86. Re-confirmed here, and it is the reason the remaining candidate is
not an obvious win.

**The binding collision is one specific pair, and it is semantic rather than statistical:**

| case | | similarity to the other-language chunk |
|---|---|---|
| `n08` | *unanswerable* — "چه سیستم‌عاملی برای اجرای گیت‌وری توصیه می‌شود؟" (which OS is recommended?) | **0.541** |
| `m05` | *answerable* — "What are the installation prerequisites and the default listen port?" | 0.525 |
| `m03` | *answerable* — which error codes cover a TLS handshake failure? | 0.523 |
| `x08` | *answerable* — what is the current version id? | 0.482 |

`n08` and `m05` are **both about installing the gateway**, in different languages, and the
nearest cross-language chunk for each is the same `deploy-guide.md` section. BGE-M3 places them
0.016 apart and puts the *unanswerable* one higher. Requiring the gate to answer `m05` and refuse
`n08` is therefore asking one embedding similarity to distinguish "which prerequisites" from
"which operating system" across a language boundary, over a chunk that is genuinely about the same
topic. That is not a gate design problem; it is the limit of what one cosine expresses.

Distinguishing them needs the question's *specific* terms matched against the candidate's text in
a shared representation. The only shared representation available without a translation resource
is a second embedding of the question against the candidate — which is the same cosine. So (c) as
specified is not implementable with what the project has, and the honest options are the two below.

**What was NOT done, deliberately.** No rule was shipped to satisfy the condition. With four
cross-lingual cases on one side of the split, any signal can be bent to pass, and a rule fitted to
four points would be documented as a capability while measuring nothing — the exact failure D57
records, where a grid's own bounds produced a confident "no feasible point" that was simply false.

**The two honest routes, for the human:**

1. **Enlarge the cross-lingual slice** before attempting (c) again — more cross-lingual *answerable*
   **and** cross-lingual *unanswerable* cases, so the condition is fitted to more than four points.
   Not guaranteed to help: if `n08`/`m05` adjacency is fundamental rather than a small-sample
   artefact, more data will show the same overlap and that is itself the answer.
2. **Accept the trade and drop cross-lingual**, documenting it as unsupported. `min_dense_alone`
   already does the safe half of this: the branch refuses everything, the gate holds **0 false
   answers on both splits**, and the README already states the limitation rather than claiming
   §2's property.

Until one of those is chosen, the branch stays as calibrated and cross-lingual stays documented as
unreliable (D61). Nothing about the current behaviour is unmeasured: `m05` and `x01` are answered,
`m03` and `x08` are refused with the chunk retrieved and semantically close, and both facts are
asserted by name in `tests/integration/test_eval_live.py`.

## D70 — the §2/§3 probe audit: one gap, and it was the dangerous one

P10 asks for "a probe test for every §2/§3 behaviour production code depends on". The audit found
almost all of them already present: the `index` permutation fallback, 401 fail-fast, 403 with the
allowed model list, 429 with and without `Retry-After`, the dimension probe, the token bucket,
L2-normalized vectors, Chroma's `n` clamping, its empty-collection behaviour, its metadata
surviving a restart, and `/ready` failing on a broken dependency. That is the finding; writing
redundant tests for them would have been noise.

The gap was the narrowest and the most dangerous. **Every limit in `config.py` was validated
relative to a constant in `config.py`** — self-consistent, and proving nothing about the provider.
Raise `MAX_CHARS_PER_ITEM` to 50 000 and every existing validation still passes, while a
45 000-character chunk that the provider rejects (the ceiling is 40 949, binary-searched over 15
probes) starts failing ingest at runtime as a 400 from a third party, visible only on long
documents.

`tests/unit/test_provider_limits.py` asserts the measured numbers, with the margins explicit rather
than merely "below": the per-item cap keeps a **2x** margin on the 40 949 ceiling; the request cap
stays under the provider's 200 000; the rate limiter stays under 120/min **and** at least 3x the
31 req/min a real re-ingest measures; and `MAX_ITEMS_PER_BATCH` is still the throughput peak (32:
29.7 chunks/s against 23.6 at 64 and 28.2 at 128), so the next reader does not raise it on a hunch.
Plus the splitter's actual invariant — no batch it builds can exceed any of the three caps — checked
on the worst case rather than on batch counts, since a planner bug pairing two oversized items would
still produce plausible counts.

One thing recorded rather than fixed: `plan_batches` raises a bare `ValueError` where §5.3 would
prefer a typed domain error. The path is **unreachable from the API** — the chunker caps a chunk at
1 050 characters against the 20 000 guard, a 19x gap — so an error class for an input nothing can
produce is the speculative work §10 rule 16 forbids. The test pins the message (the part a
reachable caller would rely on) *and* asserts the 4x gap, so a future chunker change that narrowed
it turns this into a reachable path and fails here rather than in production.

## D71 — cross-lingual retrieval is not supported, and the branch that carried it is gone

The human was offered two routes after D69 and chose **the trade**: accept that cross-lingual
retrieval is unsupported, rather than build a second signal the evaluation cannot validate. The
uncorroborated dense branch is removed and the guarantee is stated as a fact about the system
rather than about a threshold file.

```diff
- passed = (max_dense >= min_dense           AND coverage >= min_coverage)
-       OR (max_dense >= min_dense_alone)                     # uncorroborated, D47
-       OR (coverage   >= min_coverage_high)                   # exact terms
+ passed = (max_dense >= min_dense AND coverage >= min_coverage)
+       OR (coverage    >= min_coverage_high)                  # exact terms
```

**Every path into the gate now requires coverage.** So the consequence is exact and testable:
**a hit sharing no token with the question is refused at any similarity** — `max_dense` 0.99
included. `test_a_hit_sharing_no_token_with_the_question_is_refused_however_close_it_is` asserts
that across the whole similarity range, and the two live tests assert it against real vectors.

### Removing it cost nothing, and that is the finding

Re-calibrated after the removal, the metrics are **identical**:

| split | answered on answerable | false answers | gold quoted |
|---|---|---|---|
| dev (before) | 0.913 | 0 | 0.913 |
| dev (after) | 0.913 | 0 | 0.913 |
| held out (before) | 0.933 | 0 | 0.875 |
| held out (after) | 0.933 | 0 | 0.875 |

The operating point moved (`min_dense` 0.52 → 0.54, `min_coverage_high` 0.80 → 0.75) and the
outcome did not. **So the branch was not what produced the two cross-lingual answers P9 measured.**
`x01` (dense 0.618) and `m05` (0.628) were answered through *dense+coverage*, their coverage
coming from same-language documents that happen to share incidental words with the question — not
from the Persian document holding the answer. That is D61, and deleting the branch is what proved
it: the capability that a bar nobody could set safely was "retaining" was never the mechanism.

Which is the honest summary of four phases of cross-lingual work: **retrieval found the
cross-lingual chunk reliably, and every answer that reached a user got there through lexical
coincidence rather than through the embedding.**

### What is now false, and said so

`plan.md` §2 claimed cross-lingual retrieval as BGE-M3's headline property. It does not hold, the
README says it does not hold, and `config/thresholds.json` records in its own notes that
`min_dense_alone` was removed and why. The model's cross-lingual *retrieval* is real and
measured (R@3 0.95, and the correct chunk is reliably in the window). What the system does not do
is **answer** from it, and that is now a documented refusal rather than a bar that happened to
admit some cases.

### Two thresholds have now been removed, and the pattern is the point

| removed | why it had no usable value |
|---|---|
| `min_lexical` | its signal saturates at 1.0 for every query FTS matched (D58, D67) |
| `min_dense_alone` | no available signal separates a real cross-lingual hit from a near-topic one (D69, D71) |

Both were thresholds documented as evidence and measuring nothing. §10 rule 18 calls an option
with no caller "a bug waiting to be documented as a feature", and a threshold that cannot be set
to any useful value is the same failure wearing a number. A file carrying either retired key still
loads, so an existing deployment does not fail at startup over a key that no longer has a meaning.

## D72 — Swagger UI was already there, and reading what it advertised found a lie in the contract

Asked for Swagger UI "for using the endpoints". FastAPI serves `/docs`, `/redoc` and
`/openapi.json` by default and `create_app` never disabled them, so the first measurement was of
what already existed rather than of what to add:

| check | result |
|---|---|
| `GET /docs` | 200, `text/html` — served by the framework, no new dependency |
| `GET /redoc` | 200 |
| `GET /openapi.json` | 200, 8 operations |
| every operation has a `summary` | yes |
| every operation has a `description` | **no** — `GET /documents/{doc_id}` |

So the page existed and rung 1 of the ladder said the rest was optional. It was not, because
Swagger UI's whole value is that it is the contract a caller reads instead of the source, and
**the contract it was rendering was not true**:

```
POST   /documents            409  ErrorEnvelope        <- declared by hand
GET    /documents            422  HTTPValidationError  <- FastAPI's default
PUT    /documents/{doc_id}   422  HTTPValidationError
DELETE /documents/{doc_id}   422  HTTPValidationError
GET    /documents/{doc_id}   422  HTTPValidationError
```

The application has exactly one error shape: `_install_validation_error` converts every
`RequestValidationError` into `VALIDATION_ERROR` inside an `ErrorEnvelope`. So four of the twelve
declared error responses described a body this API never returns, and those four were precisely
the parameter-validating endpoints — the ones where a caller is most likely to hit a 422 and most
needs to read it. Two routes had hand-declared `422 → ErrorEnvelope`; the other four relied on
FastAPI's automatic one, and that is the whole mechanism. A caller reading the docs for a 422 body
would have written a parser against `{"detail": [...]}` and received
`{"error": {"code", "message", "request_id"}}`. **The documentation was wrong in exactly the way
that is most expensive to discover at runtime.**

Fixed by declaring it once (`INVALID_QUERY`) rather than on each of the four routes, so the
forgetting that produced the bug cannot recur silently:

- `routes.py` — `INVALID_QUERY = {422: {"model": ErrorEnvelope}}` on all four; `GET /documents/{doc_id}`
  gains the description it was the only operation missing.
- `routes.py` — `QUERY_EXAMPLES` on the `/query` body parameter. **On the parameter, not the model:**
  `json_schema_extra` on a pydantic model merges into `components.schemas`, where Swagger does not
  read it; `Body(openapi_examples=...)` is what places examples beside the schema where the UI
  reads them. Measured both ways — the model-level version left `POST /query` with no example and
  the test caught it.
- `schemas.py` — `doc_id` field example so the upload form is not half blank.
- `tests/api/test_endpoints.py` — three tests, because a fix to a contract needs something to hold
  it: the app never advertises an error shape it does not return (`HTTPValidationError` must be
  unreferenced anywhere in the schema), every operation is described and every *JSON* body is
  exemplified, and the upload field is described in a form Swagger renders as a picker.

The upload check exists because that one is third-party behaviour, and §0 rule 4 says measure it.
FastAPI 0.142 with pydantic 2.13 emits `{"type": "string", "contentMediaType":
"application/octet-stream"}` for `UploadFile` — the OpenAPI 3.2 / JSON Schema 2020-12 spelling
that replaces `format: binary`. Rather than guess whether Swagger UI honours it, the bundle
FastAPI pins was read: `swagger-ui-dist@5` implements `isFileUploadIntendedOAS32` as *true when
`contentMediaType` is a non-empty string*, so a picker **is** rendered. No override was added —
"fixing" this would have downgraded a correct 3.2 declaration to an older spelling to fix
something that was not broken. Had it been broken the failure would have been silent and total: a
text box in place of a file picker, and no document uploadable from the docs page at all.

### What this cost and what it did not

No new dependency (`fastapi` already shipped the docs routes), no vendored assets, no build step —
`swagger-ui-dist` is fetched from a CDN by the `/docs` page, so `/docs` needs network access on
the browser side while the API itself stays offline. That is worth stating rather than hiding, and
vendoring a 1.5 MB bundle into the repo to avoid it would be a worse trade.

The multipart bodies are deliberately **exempt** from the example rule: Swagger renders a file
picker for those, and a JSON example for a picker cannot be "tried". Their text fields carry
field-level examples instead. The test says so, so the exemption is a recorded decision rather than
an oversight someone will later "fix" by adding a useless example.

This is also P11's consistency pass finding its first real defect by accident: the same
docstring-versus-schema class of problem §11 lists, found by reading the schema as a user rather
than by grepping the source.

## D73 — the query LRU cache is deleted, not wired up (P11)

**Context:** `CachingEmbedder` had three cache layers: the persistent SQLite
`embedding_cache`, and in front of it a bounded in-process LRU reached through
`embed_query()`, with its own `query_lru_size` knob. P11's audit found the query path never
calls it: `RetrievalService.retrieve` reaches the embedder through `embed([question])`
(`retrieval/service.py:128`), so `embed_query` had exactly one caller in the whole repository —
its own unit test.
**Decision:** delete `embed_query`, the LRU, the `query_lru_size` parameter, and
`DEFAULT_QUERY_LRU_SIZE`. No replacement.
**Evidence:** the property the LRU existed to provide is already met by the cache that is
actually on the path. `tests/unit/test_embed_caching.py::test_a_partial_hit_embeds_only_the_misses`
proves offline that a repeated input costs no embedding request, and
`tests/integration/test_retrieval_live.py::test_a_repeated_question_costs_no_embedding_request`
proves it through the real query path against real BGE-M3 vectors. D63 measured what the LRU
would have saved: the warm-cache path is **P50 10 ms** of our own work against a **354 ms**
provider call. A third cache in front of the second bought 10 ms on a 364 ms query, at the cost
of a cache to populate, a policy to tune, a knob to document and two tests to maintain.
**Consequence:** plan.md §7.1 step 2 still reads "embed the query (LRU-cached)". That parenthetical
is now false and was left for R9's consistency pass rather than edited here, so this entry and
the plan correction land as separate commits.

### The same pass, four more deletions

`sentence_rerank` went with it, for a related reason. It had zero readers in `src` — D65 and
D1630 already record that §9.4's `SENTENCE_RERANK` has no implementation, so the field was a
promise with no code behind it, and `.env.example` advertised it as a live knob. §10 rule 18:
an option with no caller is a bug waiting to be documented as a feature. The references in
`plan.md` §7.4/§9.4 and D1060/D1630/D1774 stay, because those record that the experiment was
*considered and not run* — deleting the field is not the same as deleting the record.

Also deleted, each verified to have no reader: `_int_or_none` (`ingestion/service.py`), and three
parameters nothing read — `_install_request_id(app, settings)`, `replace_document(...,
request)`, and `_join(self, question, ...)`. `Mapping` and `Any` went with `_int_or_none`, since
it was their last consumer in that module.

### A fourth deletion the audit missed

Checking *why* `_install_request_id` did not use its `settings` argument found the reason: it
never needed one. `create_app` already calls `configure_logging(resolved)`, which installs
`SecretRedactionFilter` on the handler, so the middleware's log lines are scrubbed before they
are written. The argument was dead because the job was already done elsewhere.

Following that thread found a genuine duplicate: `api/app.py` and `logging_setup.py` each
defined `_secrets_of(settings)` — two copies of the same three lines, which is exactly the kind
of pair that drifts and leaks the token on the side that loses the race. `_secrets_of` is now
`logging_setup.secrets_of`, public, one definition, imported by `api/app.py`. I8 is a release
blocker and it is enforced by two callers of one function instead of two copies of a function.

## D74 — `docs/eval_report.md` is made real rather than repointed (P11)

**Context:** P11's audit found that `docs/eval_report.md` does not exist. It is named in
`plan.md` §7's repository layout, in §9's P9 gate, and in §11's Definition of Done — where
its box is **ticked** — and `README.md` linked to it three times. `docs/DECISIONS.md` D215
and D1654 also said `docs/eval_report.md` while D1442 said `tests/eval/report.md`, so the
project contradicted itself. The artifact that existed was `tests/eval/report.md`.
**Decision:** move the report to `docs/eval_report.md` and make the code write there.
**Evidence:** the cause was one line. `cli.py` welded the report to `EVAL_DIR` — the *eval
harness's* location — but the report is a **deliverable**, not harness input: the harness is
`dataset.jsonl` plus `runner.py`, and `runner.py` needs `EVAL_DIR` on `sys.path` for its own
import, which is a reason that has nothing to do with where the report belongs. The two are
now `EVAL_DIR` (inputs) and `REPORT_PATH` (output), which is why they drifted. `AGENTS.md`
makes `plan.md` the source of truth, and three of its statements agree with each other, so the
documentation was right and the code had drifted.

### The alternative, and why it was rejected

Repointing four documentation references at `tests/eval/report.md` is one line cheaper. It was
rejected because it makes §11's ticked box *false* — turning a documentation bug into a false
Definition-of-Done claim — and because it parks a generated deliverable inside the test tree,
where a reviewer looking for `docs/` will not find it. One line of code is the smaller truth.

### The guard, because this is D31 twice

`test_repo_consistency.py::test_every_relative_markdown_link_resolves` now walks every relative
markdown link target in `README.md`, `plan.md` and `docs/DECISIONS.md` and fails on any that
does not exist. It was written first and failed with exactly the three broken README links,
which is the only evidence that it would have caught them. The existing test explicitly
disclaimed checking path references ("which would be brittle"); this one is not brittle,
because it checks the links that exist rather than trying to parse the suite for the paths it
mentions.

It also failed on itself, once, before this entry was finished: the first draft of this
paragraph spelled the link pattern out literally in prose, and the guard read that prose as a
link to a file named `…`. That is the correct failure direction — a stricter check than
intended, not a weaker one — so the prose was reworded rather than the check loosened. Worth
recording because the tempting fix is a skip-list for "targets that do not look like paths",
which would have hidden the three real failures along with the false one.

## D75 — the embedding-cache adapter is deleted; the store *is* the cache (P11)

**Context:** `storage/embedding_cache.py` was a 27-line class whose body delegated to
`SqliteStore.get_embeddings` / `put_embeddings`, and `tests/integration/test_ingestion.py`
carried its own twelve-line copy of the same two methods — a class named `_Cache`, inside the
file whose docstring claimed "so I5 and I10 use the real table". It did use the real table,
through a duplicate of the production adapter rather than the adapter itself.
**Decision:** delete both. `SqliteStore` now satisfies the `EmbeddingCache` port structurally.
**Evidence:** the duplication was not merely redundant, it was a divergence hazard in the exact
place the release blockers are proven — if the adapter and the store ever disagreed about the
key or the model, `test_10_rebuild_restores_dense_search_with_zero_embed_calls` would still
have passed, because it was not exercising the adapter production uses. `CachingEmbedder` now
receives the store in `deps.py`, so I5 and I10 hold through the one real implementation.

### The rename had a subtlety the plan did not anticipate

`refactor-plan.md` R4 predicted renaming the port's two methods to the store's names was a
three-line change. It nearly was not, and the reason is worth recording. The port declared
`put_many(model_id, rows: Mapping[str, Sequence[float]])` while the store declares
`put_embeddings(model_id, rows: Iterable[tuple[str, Sequence[float]]])`. Those are **not** the
same type: iterating a `Mapping` yields its *keys*, not `(key, value)` pairs. The adapter's
third line, `rows.items()`, was not boilerplate — it was the conversion that made the two
signatures agree.

So the port was changed to take pairs, and `CachingEmbedder` now passes `zip(...)` directly
instead of building a dict and having the adapter take it apart again. That removes the
mismatch *and* an intermediate dict, so the lazy version was also the smaller one. The
alternative — changing `SqliteStore.put_embeddings` to accept a `Mapping` — would have touched
sixteen call sites in `tests/unit/test_store_sqlite.py` plus `reconcile.py:76`.

### Two ports, one implementation each

`reconcile.EmbedderRef` is also gone. It declared `model_id`, `dimension` and `embed()` — the
three members `domain.ports.Embedder` already declares — and existed only so a test could
hand `rebuild` a tripwire. `rebuild` now takes the real `Embedder` port, and the tripwire
satisfies it structurally. This costs the tripwire a `requests` property, because the real
port has one and the shadow did not. That is the honest price: typing an argument against the
actual contract instead of against a one-method imitation of it.

The `EmbeddingCache` port **stays**. After this change it has the store plus the in-memory
fakes in `test_embed_caching.py` — two real implementations, which is what §10 rule 16 asks
for. The fake is not decoration: `test_a_cache_repository_failure_does_not_lose_the_vectors`
needs a cache that raises, and no real store will.
