"""P7 §7.1.5: weighted RRF over the *candidate union*.

The whole point of this module's test file is the mistake D30 records: fusing
full-corpus dense ranks instead of the retrieved window makes hybrid look worse than
dense, and that looks exactly like a finding. These tests pin the union behaviour and
1-based ranking so neither can regress unnoticed.
"""

from __future__ import annotations

import pytest

from qasystem.retrieval.fusion import fuse

K = 60


def ranked(*ids: str) -> dict[str, tuple[int, float]]:
    """One arm's window: chunk_id -> (1-based rank, raw score)."""
    return {chunk_id: (rank, 1.0 - rank / 100) for rank, chunk_id in enumerate(ids, start=1)}


def score_of(fused: list[tuple[str, float]], chunk_id: str) -> float:
    return dict(fused)[chunk_id]


def test_two_empty_arms_fuse_to_nothing() -> None:
    assert fuse({}, {}, dense_weight=0.7, lexical_weight=0.3, k=K) == []


def test_a_dense_only_hit_keeps_its_dense_score() -> None:
    fused = fuse(ranked("a"), {}, dense_weight=0.7, lexical_weight=0.3, k=K)
    assert fused == [("a", 0.7 / 61)]


def test_a_lexical_only_hit_is_still_a_candidate() -> None:
    """Union, not intersection: a term only FTS found must survive into the candidate set."""
    fused = fuse({}, ranked("a"), dense_weight=0.7, lexical_weight=0.3, k=K)
    assert fused == [("a", 0.3 / 61)]


def test_a_hit_in_both_arms_scores_above_either_alone() -> None:
    both = fuse(ranked("a"), ranked("a"), dense_weight=0.7, lexical_weight=0.3, k=K)
    assert score_of(both, "a") == pytest.approx(0.7 / 61 + 0.3 / 61)


def test_ranks_are_one_based_so_the_top_hit_is_not_divided_by_k() -> None:
    """Dividing by ``k + 0`` would give the best hit the largest possible boost."""
    fused = fuse(ranked("a", "b"), {}, dense_weight=1.0, lexical_weight=0.0, k=K)
    assert score_of(fused, "a") == pytest.approx(1.0 / 61)
    assert score_of(fused, "b") == pytest.approx(1.0 / 62)


def test_k_controls_how_flat_the_curve_is() -> None:
    flat = fuse(ranked("a", "b"), {}, dense_weight=1.0, lexical_weight=0.0, k=1000)
    assert score_of(flat, "a") / score_of(flat, "b") == pytest.approx(1002 / 1001)


def test_a_deeper_rank_always_scores_lower_on_one_arm() -> None:
    fused = fuse(
        ranked(*[f"c{i}" for i in range(5)]), {}, dense_weight=1.0, lexical_weight=0.0, k=K
    )
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == 5


def test_both_arms_are_covered_not_just_the_overlap() -> None:
    fused = fuse(ranked("d1", "d2"), ranked("l1", "l2"), dense_weight=0.7, lexical_weight=0.3, k=K)
    assert {chunk_id for chunk_id, _ in fused} == {"d1", "d2", "l1", "l2"}


def test_the_result_is_sorted_best_first() -> None:
    fused = fuse(ranked("d1", "d2", "d3"), ranked("d1"), dense_weight=0.7, lexical_weight=0.3, k=K)
    assert fused[0][0] == "d1", "the hit in both arms should win"
    assert [score for _, score in fused] == sorted((s for _, s in fused), reverse=True)


def test_ties_break_deterministically_by_chunk_id() -> None:
    """Equal weights make identical mirrored scores common, so the order must not wobble."""
    dense = {"b": (1, 0.9), "a": (1, 0.9)}
    lexical = {}
    fused = fuse(dense, lexical, dense_weight=0.7, lexical_weight=0.7, k=K)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert fused == fuse(dense, lexical, dense_weight=0.7, lexical_weight=0.7, k=K)


def test_the_weights_actually_steer_the_order() -> None:
    """A rank-1 lexical hit must not be able to beat a rank-1 dense hit at 0.7/0.3."""
    fused = fuse(ranked("d"), ranked("l"), dense_weight=0.7, lexical_weight=0.3, k=K)
    assert fused[0][0] == "d"
    flipped = fuse(ranked("d"), ranked("l"), dense_weight=0.3, lexical_weight=0.7, k=K)
    assert flipped[0][0] == "l", "flipping the weights must flip the winner"


def test_a_zero_weight_arm_switches_off_entirely() -> None:
    """How §9.2 measures a single-arm baseline: the other arm's hits must not appear at all.

    Admitting them at score 0.0 would make a "lexical-only R@1" quietly measure both arms,
    and the resulting number would be wrong in a way no baseline comparison would reveal.
    """
    fused = fuse(ranked("d"), ranked("l"), dense_weight=1.0, lexical_weight=0.0, k=K)
    assert [chunk_id for chunk_id, _ in fused] == ["d"]
    assert dict(fused)["d"] == pytest.approx(1.0 / 61)

    flipped = fuse(ranked("d"), ranked("l"), dense_weight=0.0, lexical_weight=1.0, k=K)
    assert [chunk_id for chunk_id, _ in flipped] == ["l"]


def test_an_id_in_both_arms_survives_either_arm_being_off() -> None:
    fused = fuse(ranked("d"), ranked("d", "l"), dense_weight=0.0, lexical_weight=1.0, k=K)
    assert [chunk_id for chunk_id, _ in fused] == ["d", "l"]


def test_a_zero_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        fuse(ranked("a"), {}, dense_weight=1.0, lexical_weight=0.0, k=0)


def test_negative_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        fuse(ranked("a"), {}, dense_weight=-0.5, lexical_weight=0.5, k=K)
