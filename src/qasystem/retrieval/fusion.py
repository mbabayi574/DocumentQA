"""Weighted RRF over the candidate union (plan.md §7.1.5).

Reciprocal rank fusion, ``score = w_d/(k + rank_d) + w_l/(k + rank_l)``, over the **union**
of the two arms' retrieved windows -- never over full-corpus ranks.

That distinction is the whole reason this module exists. Fusing full-corpus ranks lets a
single rank-1 lexical hit outrank a correct dense hit at the very first position, which
measures a system nobody would build and looks exactly like a finding (D30). Fusing the
window means a candidate has to be *retrieved* to compete, and ranks only ever say "how
good was this hit", not "what is its absolute position in the corpus".

Ranks are 1-based. Dividing by ``k + 0`` would hand the single best hit in either arm the
largest possible boost in both, which is precisely the artefact RRF exists to remove.

Inputs are plain ``{chunk_id: (rank, raw_score)}`` maps rather than the storage adapters'
own hit types, so this stays pure: the caller supplies ranks from whichever arms it ran,
and the fusion has nothing to know about SQLite or Chroma.
"""

from __future__ import annotations

from collections.abc import Mapping


def fuse(
    dense: Mapping[str, tuple[int, float]],
    lexical: Mapping[str, tuple[int, float]],
    *,
    dense_weight: float,
    lexical_weight: float,
    k: int,
) -> list[tuple[str, float]]:
    """Fuse two ranked arms into one ``[(chunk_id, score), ...]``, best first.

    Both maps are ``chunk_id -> (1-based rank, that arm's raw score)``. The result is the
    **union**: an id only one arm returned is still a candidate, scored on that arm alone.
    The raw scores are ignored here -- they are kept for the gate, and mixing them into
    the fused score would make it impossible to tell which arm contributed what.

    A weight of **0 disables that arm entirely**, including its exclusive candidates. This
    is what makes plan §9.2's single-arm baselines real: a "lexical-only" measurement that
    still admitted dense-only hits would not be lexical-only, and the number it produced
    would silently mix two systems.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")
    if dense_weight < 0 or lexical_weight < 0:
        raise ValueError("fusion weights must not be negative")

    scores: dict[str, float] = {}
    for arm, weight in ((dense, dense_weight), (lexical, lexical_weight)):
        if not weight:
            continue  # a zero weight switches the arm off, it does not zero its contribution
        for chunk_id, (rank, _) in arm.items():
            if rank < 1:
                raise ValueError(f"ranks are 1-based, got {rank} for {chunk_id!r}")
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
    # chunk_id breaks ties so the order cannot wobble between two equal candidates --
    # which happens constantly when a single-arm baseline is measured (weight 1.0 / 0.0).
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
