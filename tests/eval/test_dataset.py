"""The eval dataset and corpus are valid (plan.md §9.1).

An eval harness is a machine for producing numbers, and a machine fed a broken input
produces confident nonsense. Every check here exists because its failure mode is a
*plausible wrong number* rather than a crash:

* A ``must_contain`` phrase that is not in the source makes R@k report a miss the system
  could never have avoided. The first live run in this project's history measured nothing
  because four of its eight questions used terms absent from the corpus, and a refusal
  looked like a finding.
* A phrase that survives the source text but is split across two chunks is the same
  failure one layer deeper: the corpus is right and the chunker decides the answer is
  unreachable. This checks the *production chunker*, so a chunking change that breaks gold
  is caught here rather than showing up as a silent recall drop.
* An unanswerable case whose discriminating term is actually in the corpus is a false
  negative waiting to happen, and it would be counted as a system defect in the report.

Absence is checked on **tokens**, not on substrings: ``طلا`` ("gold") is a substring of
``اطلاع`` ("information"), which appears in the Persian runbook. A raw ``in`` test would
have rejected a correctly unanswerable question; a tokenizer test is also the check that
matches how retrieval actually matches.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from qasystem.chunking.chunker import Chunker
from qasystem.config import load_settings
from qasystem.parsing.registry import ParserRegistry
from qasystem.text.tokenize import tokenize

from .runner import CORPUS, DATASET, Case, load_cases

#: plan.md §9.1's category counts. Stated as the plan states them.
CATEGORY_COUNTS = {
    "answerable_en": 15,
    "answerable_fa": 10,
    "exact_term": 8,
    "multi_section": 5,
    "unanswerable_near": 8,
    "unanswerable_off": 4,
}

DOCUMENTS = (
    "handbook.md",
    "runbook.txt",
    "limits.pdf",
    "deploy-guide.md",
    "incident-runbook.txt",
    "ai-engineer.pdf",
)


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    return load_cases()


@pytest.fixture(scope="module")
def settings():  # type: ignore[no-untyped-def]
    """Fake-provider settings: the harness itself never needs a token to be valid."""
    return load_settings(env_file=None, embedding_provider="fake", app_env="test")


@pytest.fixture(scope="module")
def parsed(settings) -> dict[str, str]:  # type: ignore[no-untyped-def]
    registry = ParserRegistry()
    return {
        path.name: registry.parse(path.read_bytes(), path.name).text
        for path in sorted(CORPUS.rglob("*"))
        if path.suffix in (".md", ".txt", ".pdf")
    }


@pytest.fixture(scope="module")
def chunked(settings) -> dict[str, list[str]]:  # type: ignore[no-untyped-def]
    registry, chunker = ParserRegistry(), Chunker(settings)
    out: dict[str, list[str]] = {}
    for path in sorted(CORPUS.rglob("*")):
        if path.suffix in (".md", ".txt", ".pdf"):
            document = registry.parse(path.read_bytes(), path.name)
            out[path.name] = [chunk.text for chunk in chunker.chunk(document)]
    return out


# ------------------------------------------------------------------ the corpus


def test_the_corpus_has_the_six_documents_the_plan_names() -> None:
    found = {p.name for p in CORPUS.rglob("*") if p.suffix in (".md", ".txt", ".pdf")}
    assert found == set(DOCUMENTS), sorted(found)


def test_the_corpus_covers_every_format_in_both_languages() -> None:
    """md, txt and pdf, and both languages. A format present only in one language is a
    parser bug waiting to be reported as a retrieval result."""
    extensions = {p.suffix for p in CORPUS.rglob("*") if p.suffix in (".md", ".txt", ".pdf")}
    assert extensions == {".md", ".txt", ".pdf"}
    # A PDF whose text layer is Persian is not something this machine can generate (no
    # Arabic-capable font is installed), so the plan names the committed fixture instead.
    persian_pdf = CORPUS / "fa" / "ai-engineer.pdf"
    assert persian_pdf.is_symlink() or persian_pdf.exists()
    text = persian_pdf.read_bytes()
    assert b"%PDF-" in text[:8]


def test_every_corpus_document_parses_to_a_plausible_amount_of_text(parsed) -> None:  # type: ignore[no-untyped-def]
    for name, text in parsed.items():
        assert 500 < len(text) < 6000, f"{name} parsed to {len(text)} characters"


def test_no_two_documents_state_the_same_fact() -> None:
    """The gold of an answerable case names one document. If two documents stated the
    same fact the gold would be ambiguous and the metric would punish a correct answer.

    Checked mechanically over the *gold phrases*: the same phrase must not be gold for two
    different documents anywhere in the dataset.
    """
    owners: dict[str, set[str]] = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for gold in json.loads(line)["gold"]:
            owners.setdefault(gold["must_contain"], set()).add(gold["doc"])
    shared = {phrase: sorted(docs) for phrase, docs in owners.items() if len(docs) > 1}
    assert not shared, f"gold phrase claimed by more than one document: {shared}"


# ------------------------------------------------------------------ the dataset


def test_the_dataset_meets_the_sizes_the_plan_requires(cases: list[Case]) -> None:
    assert len(cases) >= 50, len(cases)
    counts = Counter(case.category for case in cases)
    assert dict(counts) == CATEGORY_COUNTS, dict(counts)


def test_the_split_is_sixty_forty_and_every_category_reaches_both_splits(
    cases: list[Case],
) -> None:
    dev = sum(1 for case in cases if case.split == "dev")
    assert 0.55 <= dev / len(cases) <= 0.65, (dev, len(cases))
    by_category: dict[str, set[str]] = {}
    for case in cases:
        by_category.setdefault(case.category, set()).add(case.split)
    empty = {c: s for c, s in by_category.items() if s != {"dev", "test"}}
    assert not empty, f"a category with no held-out case measures nothing held out: {empty}"


def test_every_answerable_case_has_gold_and_every_unanswerable_case_has_none(
    cases: list[Case],
) -> None:
    for case in cases:
        if case.category.startswith("unanswerable"):
            assert not case.gold, case.id
        else:
            assert case.gold, case.id
            assert all(g.doc in DOCUMENTS and g.must_contain.strip() for g in case.gold), case.id


def test_multi_section_cases_ask_for_more_than_one_fact(cases: list[Case]) -> None:
    """A "multi-section" case with one gold phrase is an ordinary case with a misleading
    label, and it would make the multi_section count a fiction."""
    for case in cases:
        if case.category == "multi_section":
            assert len(case.gold) >= 2, case.id


def test_every_gold_phrase_is_verbatim_in_the_source_text(
    cases: list[Case], parsed: dict[str, str]
) -> None:
    """The first live run in this project measured nothing: four of its eight questions used
    terms that appear in no document, and the gate's correct refusal read as a defect."""
    missing = [
        (case.id, gold.doc, gold.must_contain)
        for case in cases
        for gold in case.gold
        if gold.must_contain not in parsed[gold.doc]
    ]
    assert not missing, missing


def test_every_gold_phrase_survives_the_production_chunker(
    cases: list[Case], chunked: dict[str, list[str]]
) -> None:
    """The source is right but the chunker split the phrase, so the fact is unreachable at
    any recall. Checked against the real chunker, so a chunking change that breaks gold
    fails here instead of appearing later as a silent recall drop."""
    split = [
        (case.id, gold.doc, gold.must_contain)
        for case in cases
        for gold in case.gold
        if not any(gold.must_contain in text for text in chunked[gold.doc])
    ]
    assert not split, split


def test_unanswerable_cases_declare_a_discriminating_term(cases: list[Case]) -> None:
    for case in cases:
        if not case.answerable:
            assert case.absent, f"{case.id} does not say what would make it answerable"


def test_no_unanswerable_case_is_actually_answerable(
    cases: list[Case], parsed: dict[str, str]
) -> None:
    """Token-based, not substring-based: ``طلا`` is a substring of ``اطلاع`` ("information"),
    which is in the Persian runbook, and a raw ``in`` test would reject a correct case."""
    corpus_tokens = set()
    for text in parsed.values():
        corpus_tokens.update(tokenize(text))
    for case in cases:
        if case.answerable or not case.absent:
            continue
        terms = list(dict.fromkeys(tokenize(case.absent)))
        present = [term for term in terms if term in corpus_tokens]
        assert len(present) < len(terms), (
            f"{case.id}: {case.absent!r} is entirely present as corpus tokens ({present}), "
            "so the question is answerable and would be counted as a system defect"
        )


def test_the_eval_corpus_includes_cross_lingual_cases(cases: list[Case]) -> None:
    """Cross-lingual retrieval is the capability BGE-M3 is chosen for (§2) and the one the
    gate's uncorroborated branch exists for (D47). A corpus that never crosses languages
    would let the whole system be calibrated without ever testing it."""
    from qasystem.text.language import detect_language

    persian_docs = {"deploy-guide.md", "incident-runbook.txt", "ai-engineer.pdf"}
    crossed = [
        case.id
        for case in cases
        if case.answerable
        and detect_language(case.question) != ("fa" if case.gold[0].doc in persian_docs else "en")
    ]
    assert len(crossed) >= 3, (
        f"only {len(crossed)} cross-lingual answerable cases ({crossed}); the headline "
        "capability is then measured on too little to mean anything"
    )


def test_case_ids_are_unique(cases: list[Case]) -> None:
    ids = Counter(case.id for case in cases)
    assert [name for name, count in ids.items() if count > 1] == []
