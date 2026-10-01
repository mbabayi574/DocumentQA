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
