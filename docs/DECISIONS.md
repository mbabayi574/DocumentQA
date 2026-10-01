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
