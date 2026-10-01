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
