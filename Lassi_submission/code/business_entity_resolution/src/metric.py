"""Macro-averaged F-beta scorer for the Business Entity Resolution challenge.

This is the single source of truth for "how good is a submission". Every
threshold, feature and model decision in this pipeline is measured against
`macro_f_beta`, so correctness here matters more than anywhere else in the
codebase -- a subtly wrong metric silently optimises the wrong objective for
the entire event.

The challenge definition (beta = 0.5, precision-weighted):

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed **per Source-1 entity**, then averaged across *all* Source-1 entities
in the evaluation set -- a macro average, not a pooled/micro one.

Singleton convention, quoted from the problem statement:

    "A Source 1 entity with no true matches scores 1.0 when you correctly
    predict an empty list, and 0.0 when you predict any match for it."

That makes the per-entity score discontinuous, and it is the reason the
abstain decision is a first-class part of the pipeline rather than a
post-processing detail.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

# Challenge metric. Kept as a named constant so no caller has to guess.
DEFAULT_BETA = 0.5

EntitySets = Mapping[str, Iterable[str]]


def f_beta(precision: float, recall: float, beta: float = DEFAULT_BETA) -> float:
    """F-beta from precision and recall.

    `beta` weights recall `beta` times as much as precision, so beta < 1 is
    precision-heavy. Returns 0.0 when both inputs are 0 (the degenerate case
    where a prediction and a truth set share nothing).
    """
    b2 = beta * beta
    denominator = b2 * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denominator


def entity_f_beta(
    predicted: Iterable[str],
    truth: Iterable[str],
    beta: float = DEFAULT_BETA,
) -> float:
    """Score a single Source-1 entity.

    Handles the four cases explicitly rather than relying on arithmetic to fall
    out correctly, because two of them are 0/0 divisions:

    ==================  ==================  =======
    truth               predicted           score
    ==================  ==================  =======
    empty               empty               1.0
    empty               non-empty           0.0
    non-empty           empty               0.0   (recall = 0)
    non-empty           non-empty           F-beta
    ==================  ==================  =======
    """
    pred_set = predicted if isinstance(predicted, (set, frozenset)) else set(predicted)
    true_set = truth if isinstance(truth, (set, frozenset)) else set(truth)

    if not true_set:
        # Correctly abstaining on a singleton is a full point; guessing is a zero.
        return 1.0 if not pred_set else 0.0
    if not pred_set:
        # Recall is 0, so F-beta is 0 regardless of beta.
        return 0.0

    overlap = len(pred_set & true_set)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall = overlap / len(true_set)
    return f_beta(precision, recall, beta)


def macro_f_beta(
    predictions: EntitySets,
    truths: EntitySets,
    beta: float = DEFAULT_BETA,
    strict: bool = True,
) -> float:
    """Macro-average `entity_f_beta` over every entity in `truths`.

    `truths` defines the evaluation set: the average runs over its keys, so an
    entity missing from `predictions` scores as an empty prediction. That
    mirrors the real scorer, where a missing row is a rejection -- hence
    `strict=True` raises instead of silently scoring it.

    Args:
        predictions: entity id -> predicted match ids.
        truths: entity id -> true match ids. Defines the denominator.
        beta: metric beta. Defaults to the challenge's 0.5.
        strict: raise if `predictions` omits a truth key or adds unknown keys.

    Returns:
        Mean per-entity F-beta in [0, 1]. Returns 0.0 for an empty truth set.
    """
    if not truths:
        logger.warning("macro_f_beta called with an empty truth set; returning 0.0")
        return 0.0

    if strict:
        missing = truths.keys() - predictions.keys()
        if missing:
            raise ValueError(
                f"{len(missing)} entity(ies) in truths have no prediction, "
                f"e.g. {sorted(missing)[:5]}. The real scorer rejects a submission "
                f"with missing rows, so this is an error rather than a zero."
            )
        extra = predictions.keys() - truths.keys()
        if extra:
            raise ValueError(
                f"{len(extra)} prediction(s) reference unknown entity ids, "
                f"e.g. {sorted(extra)[:5]}."
            )

    total = 0.0
    for entity_id, true_ids in truths.items():
        total += entity_f_beta(predictions.get(entity_id, ()), true_ids, beta)
    return total / len(truths)


def score_breakdown(
    predictions: EntitySets,
    truths: EntitySets,
    beta: float = DEFAULT_BETA,
) -> dict[str, float]:
    """Split the macro score into its singleton and non-singleton parts.

    The two behave very differently -- singletons are all-or-nothing while
    non-singletons are graded -- so a single aggregate number hides which half
    a change actually moved. Used for diagnostics, never for the headline score.
    """
    singleton_total = singleton_n = 0.0
    multi_total = multi_n = 0.0

    for entity_id, true_ids in truths.items():
        score = entity_f_beta(predictions.get(entity_id, ()), true_ids, beta)
        if not true_ids:
            singleton_total += score
            singleton_n += 1
        else:
            multi_total += score
            multi_n += 1

    n = singleton_n + multi_n
    return {
        "macro_f_beta": (singleton_total + multi_total) / n if n else 0.0,
        "singleton_mean": singleton_total / singleton_n if singleton_n else float("nan"),
        "singleton_count": singleton_n,
        "singleton_share": singleton_n / n if n else 0.0,
        # How much of the final score each half contributes, in absolute points.
        "singleton_contribution": singleton_total / n if n else 0.0,
        "non_singleton_mean": multi_total / multi_n if multi_n else float("nan"),
        "non_singleton_count": multi_n,
        "non_singleton_contribution": multi_total / n if n else 0.0,
    }
