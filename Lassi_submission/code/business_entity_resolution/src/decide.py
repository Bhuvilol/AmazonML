"""Decision stage: turn pair probabilities into the per-entity output set.

This is the stage most competitors will under-build, and it is where the
measured properties of this challenge pay off. Three facts drive it:

1. **The metric is macro-averaged per entity, weighting precision 2x.**
   Concretely, for an entity with 4 true matches: emitting 3 correct scores
   0.9375, while emitting all 4 plus one false positive scores 0.8333.
   **Missing a true match beats adding a wrong one.** The threshold therefore
   belongs well above 0.5, and it is tuned directly against macro F_0.5 rather
   than against any pairwise proxy.

2. **Singletons score 1.0 or 0.0, nothing between.** They are 5.58% of training
   entities. That share is small enough that abstention is a guard rather than
   the main lever -- an earlier draft of this pipeline over-weighted it -- but
   a wrong emission on a singleton still forfeits that entity entirely.

3. **Matches are mutually exclusive.** Verified on the full training ground
   truth: 7,638,365 distinct Source-2/3 ids, and **zero** claimed by more than
   one Source-1 entity. So if two entities both want the same record, at most
   one is right, and dropping the weaker claim is a strict precision gain --
   which a precision-weighted metric rewards twice over. This constraint is
   free, exact, and most entrants will not check for it.

The stages are applied in order and each is measured against the previous;
anything that does not pay for itself gets removed rather than kept for
elegance.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def resolve_exclusivity(
    rows: np.ndarray,
    cols: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Keep, for each target record, only its single highest-scoring claim.

    Enforces the verified constraint that a Source-2/3 record belongs to at
    most one Source-1 entity. Fully vectorised: one lexsort plus a boundary
    scan, so it costs milliseconds even at tens of millions of pairs.

    Returns:
        Boolean mask over the input arrays marking the claims to keep.
    """
    if len(rows) == 0:
        return np.zeros(0, dtype=bool)

    # Sort by target id, then by descending score, so each target's best claim
    # is the first entry in its run.
    order = np.lexsort((-scores, cols))
    sorted_cols = cols[order]

    is_first = np.empty(len(order), dtype=bool)
    is_first[0] = True
    np.not_equal(sorted_cols[1:], sorted_cols[:-1], out=is_first[1:])

    keep = np.zeros(len(rows), dtype=bool)
    keep[order[is_first]] = True

    dropped = len(rows) - int(keep.sum())
    if dropped:
        logger.info(
            "exclusivity: dropped %d contested claim(s) of %d (%.2f%%)",
            dropped, len(rows), 100 * dropped / len(rows),
        )
    return keep


def select(
    rows: np.ndarray,
    cols: np.ndarray,
    probabilities: np.ndarray,
    n_entities: int,
    threshold: float,
    singleton_gate: float | None = None,
    enforce_exclusivity: bool = True,
) -> list[np.ndarray]:
    """Choose each entity's predicted match set.

    Args:
        rows: source-entity index per candidate pair.
        cols: target-record index per candidate pair.
        probabilities: model P(match) per candidate pair.
        n_entities: total source entities (output has one entry each).
        threshold: emit pairs at or above this probability.
        singleton_gate: if set, an entity whose *best* candidate falls below
            this value emits nothing at all. A second, stricter bar than
            ``threshold`` for committing to an entity having any match --
            useful only when it exceeds ``threshold``.
        enforce_exclusivity: apply :func:`resolve_exclusivity`.

    Returns:
        One array of target indices per entity, in entity order. Empty arrays
        are singleton predictions and must still be written as rows.
    """
    keep = probabilities >= threshold

    if singleton_gate is not None and singleton_gate > threshold:
        # Best probability per entity, computed over all candidates.
        best = np.zeros(n_entities, dtype=np.float32)
        np.maximum.at(best, rows, probabilities)
        keep &= best[rows] >= singleton_gate

    sel_rows, sel_cols, sel_p = rows[keep], cols[keep], probabilities[keep]

    if enforce_exclusivity and len(sel_rows):
        winners = resolve_exclusivity(sel_rows, sel_cols, sel_p)
        sel_rows, sel_cols = sel_rows[winners], sel_cols[winners]

    # Bucket by entity. np.split on a sorted array is far faster than a
    # per-entity Python loop at 1.7M entities.
    order = np.argsort(sel_rows, kind="stable")
    sel_rows, sel_cols = sel_rows[order], sel_cols[order]
    counts = np.bincount(sel_rows, minlength=n_entities)
    boundaries = np.cumsum(counts)[:-1]
    return np.split(sel_cols, boundaries)


def macro_f05_from_indices(
    predicted: list[np.ndarray],
    truth: list[set],
) -> float:
    """Macro F_0.5 over index sets -- the same definition as metric.py.

    Kept here operating on integer indices so the threshold sweep never has to
    materialise id strings, which at 65M pairs would dominate its runtime.
    """
    total = 0.0
    for pred, true_set in zip(predicted, truth):
        if not true_set:
            total += 1.0 if len(pred) == 0 else 0.0
            continue
        if len(pred) == 0:
            continue
        overlap = len(true_set.intersection(pred.tolist()))
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(true_set)
        total += 1.25 * precision * recall / (0.25 * precision + recall)
    return total / len(truth) if truth else 0.0


def sweep_threshold(
    rows: np.ndarray,
    cols: np.ndarray,
    probabilities: np.ndarray,
    truth: list[set],
    thresholds: np.ndarray | None = None,
    singleton_gate: float | None = None,
    enforce_exclusivity: bool = True,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Find the threshold maximising macro F_0.5 on a validation split.

    This is tuned on local validation and never on the public leaderboard:
    final rankings come from the private split, the metric is macro-averaged
    over a subset and therefore noisy, and the threshold is the largest single
    lever in the pipeline. Tuning the biggest lever against the noisiest signal
    is how a strong model loses on the private board.

    Returns:
        ``(best_threshold, best_score, full_curve)``.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.025), 4)

    n_entities = len(truth)
    curve: list[tuple[float, float]] = []
    for value in thresholds:
        predicted = select(
            rows, cols, probabilities, n_entities,
            threshold=float(value),
            singleton_gate=singleton_gate,
            enforce_exclusivity=enforce_exclusivity,
        )
        curve.append((float(value), macro_f05_from_indices(predicted, truth)))

    best_threshold, best_score = max(curve, key=lambda kv: kv[1])
    logger.info("best threshold %.3f -> macro F0.5 %.4f", best_threshold, best_score)
    return best_threshold, best_score, curve
