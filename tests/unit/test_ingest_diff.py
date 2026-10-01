"""P6: the pure chunk diff. No I/O, so every multiset case is cheap to pin.

The diff exists to answer one question for the ingest log: how many chunks are new,
reused, and gone. It never decides what to embed -- that is the embedding cache's job,
keyed on the exact embedded string. Keeping it pure is what makes plan test 12 (a
reorder-only edit costs nothing) directly assertable.
"""

from __future__ import annotations

from qasystem.ingestion.diff import diff_hashes


def test_an_unchanged_document_differs_in_nothing() -> None:
    plan = diff_hashes(["a", "b", "c"], ["a", "b", "c"])
    assert plan.added == ()
    assert plan.reused == (0, 1, 2)
    assert plan.removed == ()


def test_a_first_ingest_adds_everything() -> None:
    plan = diff_hashes([], ["a", "b"])
    assert plan.added == (0, 1)
    assert plan.reused == ()
    assert plan.removed == ()
    assert plan.chunks_added == 2 and plan.chunks_reused == 0


def test_a_local_edit_adds_only_the_new_chunk() -> None:
    """Plan test 3: a local edit must re-embed nothing but the paragraph that changed."""
    plan = diff_hashes(["a", "b", "c"], ["a", "b2", "c"])
    assert plan.added == (1,)
    assert plan.reused == (0, 2)
    assert plan.removed == ("b",)


def test_a_reorder_only_edit_reuses_everything() -> None:
    """Plan test 12: moving a section must not cost an embedding call."""
    plan = diff_hashes(["a", "b", "c"], ["c", "a", "b"])
    assert plan.added == ()
    assert plan.reused == (0, 1, 2)
    assert plan.removed == ()


def test_duplicate_chunks_keep_their_multiplicity() -> None:
    """Plan test 12: two identical chunks are two chunks, not one."""
    plan = diff_hashes(["a", "a", "b"], ["a", "a", "b"])
    assert plan.reused == (0, 1, 2)
    assert plan.added == ()

    # One of three copies survives an edit, so two are genuinely new.
    grew = diff_hashes(["a"], ["a", "a", "a"])
    assert grew.added == (1, 2)
    assert grew.reused == (0,)

    # And one copy disappearing is a real removal, not a no-op.
    shrank = diff_hashes(["a", "a", "a"], ["a"])
    assert shrank.added == ()
    assert shrank.reused == (0,)
    assert shrank.removed == ("a", "a")


def test_removed_chunks_are_reported_by_hash() -> None:
    plan = diff_hashes(["a", "b", "c"], ["a"])
    assert plan.removed == ("b", "c")
    assert plan.added == ()


def test_appended_content_adds_only_the_tail() -> None:
    plan = diff_hashes(["a"], ["a", "b", "c"])
    assert plan.added == (1, 2)
    assert plan.reused == (0,)


def test_the_ordering_of_reused_follows_the_new_document() -> None:
    """Reused positions are indices into the *new* chunk list, which is what stage needs."""
    plan = diff_hashes(["x", "a", "y"], ["a", "y"])
    assert plan.added == ()
    assert plan.reused == (0, 1)
    assert plan.removed == ("x",)


def test_counts_are_available_without_walking_the_tuples() -> None:
    plan = diff_hashes(["a", "b"], ["a", "b", "c"])  # appended, nothing dropped
    assert (plan.chunks_added, plan.chunks_reused, plan.chunks_removed) == (1, 2, 0)

    trimmed = diff_hashes(["a", "b", "c"], ["a", "b", "c", "d", "e"])
    assert (trimmed.chunks_added, trimmed.chunks_reused, trimmed.chunks_removed) == (2, 3, 0)
    assert plan.total_new == 3


def test_two_empty_lists_diff_to_nothing() -> None:
    plan = diff_hashes([], [])
    assert plan.total_new == 0
    assert plan.is_empty_change is True


def test_removed_keeps_one_copy_per_missing_occurrence() -> None:
    plan = diff_hashes(["a", "a"], ["a"])
    assert plan.removed == ("a",)
    assert plan.chunks_removed == 1
