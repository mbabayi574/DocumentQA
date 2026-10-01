"""P3: deterministic chunking — the tests plan.md P3 lists, plus the invariants
that make I6 mechanically checkable."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from qasystem.chunking.chunker import CHARS_PER_TOKEN, Chunker, split_sentences
from qasystem.config import load_settings
from qasystem.parsing.registry import ParserRegistry

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "docs"


@pytest.fixture
def chunker() -> Chunker:
    settings = load_settings(env_file=None, embedding_provider="fake", app_env="test")
    return Chunker(settings)


@pytest.fixture
def parser() -> ParserRegistry:
    return ParserRegistry(max_upload_mb=20)


def md(body: str, name: str = "doc.md"):

    from qasystem.parsing.markdown import MarkdownParser

    return MarkdownParser().parse(body.encode(), name)


# --- the five P3 tests -------------------------------------------------------


def test_chunking_is_deterministic(chunker: Chunker, parser: ParserRegistry) -> None:
    doc = parser.parse((FIXTURES / "en" / "storyen.md").read_bytes(), "storyen.md")
    first = chunker.chunk(doc)
    second = chunker.chunk(doc)
    assert [c.chunk_hash for c in first] == [c.chunk_hash for c in second]
    assert [(c.char_start, c.char_end, c.text) for c in first] == [
        (c.char_start, c.char_end, c.text) for c in second
    ]


def test_every_source_paragraph_is_covered(chunker: Chunker, parser: ParserRegistry) -> None:
    doc = parser.parse((FIXTURES / "en" / "storyen.md").read_bytes(), "storyen.md")
    chunks = chunker.chunk(doc)
    assert chunks
    for section in doc.sections:
        covered = any(
            chunk.char_start < section.char_end and chunk.char_end > section.char_start
            for chunk in chunks
        )
        assert covered, f"section {section.section_path} is not covered by any chunk"


def test_no_chunk_exceeds_the_hard_cap(chunker: Chunker, parser: ParserRegistry) -> None:
    doc = parser.parse((FIXTURES / "en" / "clean-code-excerpt.pdf").read_bytes(), "book.pdf")
    hard = chunker.hard_max_chars
    oversized = [c for c in chunker.chunk(doc) if len(c.text) > hard]
    assert not oversized, f"{len(oversized)} chunks exceed {hard} chars"


def test_chunk_text_is_an_exact_slice_of_the_document(
    chunker: Chunker, parser: ParserRegistry
) -> None:
    """I6: this is the property that makes citations verifiable."""
    for name in ["en/storyen.md", "fa/storyfa.md", "en/clean-code-excerpt.pdf"]:
        doc = parser.parse((FIXTURES / name).read_bytes(), Path(name).name)
        for chunk in chunker.chunk(doc):
            assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


def test_local_paragraph_edit_preserves_most_chunk_hashes(
    chunker: Chunker, parser: ParserRegistry
) -> None:
    """Plan.md P3: editing one paragraph must not re-hash the whole document."""
    paragraphs = [f"Paragraph {i} describes topic {i} in some detail. " * 3 for i in range(12)]
    original = md("\n\n".join(paragraphs))
    before = chunker.chunk(original)

    edited_paragraphs = list(paragraphs)
    edited_paragraphs[5] = "Paragraph 5 now describes a completely different subject."
    after = chunker.chunk(md("\n\n".join(edited_paragraphs)))

    changed_span = (
        original.text.index(paragraphs[5]),
        original.text.index(paragraphs[5]) + len(paragraphs[5]),
    )
    untouched_before = {
        c.chunk_hash
        for c in before
        if not (c.char_start < changed_span[1] and c.char_end > changed_span[0])
    }
    untouched_after = {
        c.chunk_hash
        for c in after
        if not (c.char_start < changed_span[1] and c.char_end > changed_span[0])
    }
    if not untouched_before:
        pytest.skip("fixture produced only one chunk; nothing to compare")
    kept = untouched_before & untouched_after
    assert len(kept) / len(untouched_before) >= 0.8


# --- structural guarantees ---------------------------------------------------


def test_unrelated_top_level_sections_are_never_merged(
    chunker: Chunker,
) -> None:
    doc = md("# Alpha\n\n" + "Alpha sentence. " * 5 + "\n\n# Beta\n\n" + "Beta sentence. " * 5)
    chunks = chunker.chunk(doc)
    alpha = [c for c in chunks if c.section_path[0] == "Alpha"]
    beta = [c for c in chunks if c.section_path[0] == "Beta"]
    assert alpha and beta
    assert all(c.section_path == ("Alpha",) for c in alpha)
    assert all(c.section_path == ("Beta",) for c in beta)


def test_ordinals_are_dense_and_ordered(chunker: Chunker, parser: ParserRegistry) -> None:
    doc = parser.parse((FIXTURES / "fa" / "storyfa.md").read_bytes(), "storyfa.md")
    chunks = chunker.chunk(doc)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(a.char_start <= b.char_start for a, b in pairwise(chunks))


def test_language_is_recorded_per_chunk(chunker: Chunker, parser: ParserRegistry) -> None:
    doc = parser.parse((FIXTURES / "fa" / "storyfa.md").read_bytes(), "storyfa.md")
    assert all(chunk.language == "fa" for chunk in chunker.chunk(doc))


def test_page_numbers_survive_for_pdf_sources(chunker: Chunker, parser: ParserRegistry) -> None:
    doc = parser.parse((FIXTURES / "fa" / "ai-engineer.pdf").read_bytes(), "ai.pdf")
    chunks = chunker.chunk(doc)
    assert chunks
    assert all(chunk.page_start is not None for chunk in chunks)
    assert min(c.page_start for c in chunks) >= 1  # type: ignore[arg-type]


def test_chars_per_token_is_the_measured_conservative_value() -> None:
    """plan.md 2.1/§0.5: sized against the densest measured text, not prose."""
    assert CHARS_PER_TOKEN == 1.5


# --- sentence splitting ------------------------------------------------------


def test_sentences_split_on_terminators() -> None:
    spans = split_sentences("First one. Second one! Third one?")
    assert [span.text for span in spans] == ["First one.", "Second one!", "Third one?"]


def test_persian_terminators_split() -> None:
    spans = split_sentences("جمله اول. جمله دوم! پرسش سوم؟")
    assert len(spans) == 3


def test_decimals_do_not_split() -> None:
    spans = split_sentences("The ratio is 3.14 exactly.")
    assert [s.text for s in spans] == ["The ratio is 3.14 exactly."]


def test_abbreviations_do_not_split() -> None:
    spans = split_sentences("Dr. Smith went home. Then he slept.")
    assert len(spans) == 2


def test_sentence_offsets_index_the_input() -> None:
    text = "Alpha here. Beta there."
    for span in split_sentences(text):
        assert text[span.start : span.end] == span.text
