"""Tests for the challenge metric.

The headline test is `test_problem_statement_example`: the problem statement
gives one fully worked example with an expected value of 0.714, and that is the
only external ground truth available for the scorer. Everything else in this
pipeline is calibrated against this module, so it is tested first and hardest.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metric import (  # noqa: E402
    DEFAULT_BETA,
    entity_f_beta,
    f_beta,
    macro_f_beta,
    score_breakdown,
)


# --------------------------------------------------------------------------
# The one externally-specified value
# --------------------------------------------------------------------------

def test_problem_statement_example():
    """PS: predicted 3 ids, truth 2 of them -> P=2/3, R=1.0, F_0.5 = 0.714."""
    predicted = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}

    score = entity_f_beta(predicted, truth)

    assert score == pytest.approx(0.714, abs=5e-4)
    # Exact value, to guard against a "close enough" regression.
    assert score == pytest.approx(0.8333333333 / 1.1666666667, rel=1e-9)


def test_beta_default_is_challenge_value():
    assert DEFAULT_BETA == 0.5


# --------------------------------------------------------------------------
# The four cases that involve a 0/0 division
# --------------------------------------------------------------------------

def test_singleton_correctly_abstained_scores_one():
    assert entity_f_beta(set(), set()) == 1.0


def test_singleton_with_any_prediction_scores_zero():
    assert entity_f_beta({"S2-1"}, set()) == 0.0
    # Still zero no matter how many are predicted.
    assert entity_f_beta({"S2-1", "S3-2", "S2-3"}, set()) == 0.0


def test_empty_prediction_on_real_matches_scores_zero():
    assert entity_f_beta(set(), {"S2-1"}) == 0.0


def test_disjoint_sets_score_zero():
    assert entity_f_beta({"S2-9"}, {"S2-1"}) == 0.0


def test_perfect_prediction_scores_one():
    ids = {"S2-1", "S3-2"}
    assert entity_f_beta(ids, ids) == 1.0


# --------------------------------------------------------------------------
# The precision asymmetry -- these numbers drive the whole strategy
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "predicted, truth, expected, note",
    [
        ({"a", "b"}, {"a", "b"}, 1.0, "2-match: both correct"),
        ({"a", "b", "x"}, {"a", "b"}, 0.7142857, "2-match: both + 1 false positive"),
        ({"a"}, {"a", "b"}, 0.8333333, "2-match: only 1 of 2, no false positive"),
        ({"a"}, {"a"}, 1.0, "1-match: correct"),
        ({"a", "x"}, {"a"}, 0.5555556, "1-match: correct + 1 false positive"),
        ({"a", "b", "c"}, {"a", "b", "c", "d"}, 0.9375, "4-match: 3 of 4, no FP"),
        ({"a", "b", "c", "d", "x"}, {"a", "b", "c", "d"}, 0.8333333, "4-match: all 4 + 1 FP"),
    ],
)
def test_known_asymmetry_values(predicted, truth, expected, note):
    assert entity_f_beta(predicted, truth) == pytest.approx(expected, abs=1e-6), note


def test_missing_a_match_beats_adding_a_false_positive():
    """The dominant gradient of the problem, asserted rather than assumed.

    At the real data's modal cardinality (3-4 matches), dropping a true match
    scores strictly better than emitting everything plus one wrong id. This is
    why the decision stage is precision-biased.
    """
    truth = {"a", "b", "c", "d"}

    miss_one = entity_f_beta({"a", "b", "c"}, truth)
    all_plus_one_fp = entity_f_beta({"a", "b", "c", "d", "x"}, truth)

    assert miss_one > all_plus_one_fp
    assert miss_one == pytest.approx(0.9375)
    assert all_plus_one_fp == pytest.approx(0.8333333, abs=1e-6)


def test_precision_weighted_more_than_recall():
    """beta=0.5 must favour precision: swapping P and R should not be symmetric."""
    high_precision = entity_f_beta({"a"}, {"a", "b"})          # P=1.0, R=0.5
    high_recall = entity_f_beta({"a", "b"}, {"a"})             # P=0.5, R=1.0
    assert high_precision > high_recall


# --------------------------------------------------------------------------
# f_beta primitive
# --------------------------------------------------------------------------

def test_f_beta_zero_when_both_inputs_zero():
    assert f_beta(0.0, 0.0) == 0.0


def test_f_beta_equals_f1_when_beta_is_one():
    p, r = 0.6, 0.4
    expected_f1 = 2 * p * r / (p + r)
    assert f_beta(p, r, beta=1.0) == pytest.approx(expected_f1)


# --------------------------------------------------------------------------
# Macro averaging
# --------------------------------------------------------------------------

def test_macro_averages_over_entities_equally():
    """Every entity counts the same, regardless of how many matches it has."""
    truths = {
        "S1-1": {"a"},                       # 1 match
        "S1-2": {"a", "b", "c", "d", "e"},   # 5 matches
    }
    predictions = {
        "S1-1": {"a"},                       # perfect -> 1.0
        "S1-2": set(),                       # empty    -> 0.0
    }
    # Macro: (1.0 + 0.0) / 2. A micro/pooled metric would weight S1-2 heavily.
    assert macro_f_beta(predictions, truths) == pytest.approx(0.5)


def test_all_empty_submission_scores_the_singleton_rate():
    """The trivial floor: predicting nothing scores exactly the singleton share.

    Measured on the real training ground truth this is 0.0558, which is why the
    singleton gate is a guard rather than the main lever.
    """
    truths = {f"S1-{i}": set() for i in range(5)}              # 5 singletons
    truths.update({f"S1-{i}": {"a"} for i in range(5, 100)})   # 95 with matches
    all_empty = {k: set() for k in truths}

    assert macro_f_beta(all_empty, truths) == pytest.approx(0.05)


def test_strict_mode_rejects_missing_predictions():
    truths = {"S1-1": {"a"}, "S1-2": set()}
    with pytest.raises(ValueError, match="no prediction"):
        macro_f_beta({"S1-1": {"a"}}, truths)


def test_strict_mode_rejects_unknown_entities():
    truths = {"S1-1": {"a"}}
    with pytest.raises(ValueError, match="unknown entity"):
        macro_f_beta({"S1-1": {"a"}, "S1-99": {"b"}}, truths)


def test_non_strict_mode_treats_missing_as_empty():
    truths = {"S1-1": {"a"}, "S1-2": set()}
    # S1-1 missing -> 0.0; S1-2 missing -> empty prediction on a singleton -> 1.0
    assert macro_f_beta({}, truths, strict=False) == pytest.approx(0.5)


def test_empty_truths_returns_zero():
    assert macro_f_beta({}, {}) == 0.0


def test_accepts_lists_not_just_sets():
    assert entity_f_beta(["a", "b"], ["a", "b"]) == 1.0


# --------------------------------------------------------------------------
# Diagnostic breakdown
# --------------------------------------------------------------------------

def test_score_breakdown_splits_contributions():
    truths = {
        "S1-1": set(),            # singleton, predicted empty -> 1.0
        "S1-2": set(),            # singleton, predicted wrong -> 0.0
        "S1-3": {"a", "b"},       # perfect -> 1.0
        "S1-4": {"a", "b"},       # half    -> 0.8333
    }
    predictions = {
        "S1-1": set(),
        "S1-2": {"x"},
        "S1-3": {"a", "b"},
        "S1-4": {"a"},
    }

    out = score_breakdown(predictions, truths)

    assert out["singleton_count"] == 2
    assert out["non_singleton_count"] == 2
    assert out["singleton_share"] == pytest.approx(0.5)
    assert out["singleton_mean"] == pytest.approx(0.5)
    assert out["non_singleton_mean"] == pytest.approx((1.0 + 0.8333333) / 2, abs=1e-6)
    # Contributions must sum to the headline score.
    assert out["singleton_contribution"] + out["non_singleton_contribution"] == pytest.approx(
        out["macro_f_beta"]
    )
    assert out["macro_f_beta"] == pytest.approx(macro_f_beta(predictions, truths))


def test_breakdown_handles_no_singletons():
    truths = {"S1-1": {"a"}}
    out = score_breakdown({"S1-1": {"a"}}, truths)
    assert math.isnan(out["singleton_mean"])
    assert out["singleton_count"] == 0


# --------------------------------------------------------------------------
# Regression guard: predictions and truth must share one index space
# --------------------------------------------------------------------------

def test_offset_predictions_against_unoffset_truth_scores_zero():
    """Pins the bug that made a multi-country run report 0.469 instead of 0.93.

    Candidate columns were offset to make them globally unique across country
    partitions, but the ground-truth sets were not offset to match. Every
    intersection for the second partition was then empty, zeroing that whole
    country. This asserts the failure mode so it cannot return silently.
    """
    truth = {10, 20}
    offset = 1_000_000

    aligned = entity_f_beta({10, 20}, truth)
    misaligned = entity_f_beta({10 + offset, 20 + offset}, truth)

    assert aligned == 1.0
    assert misaligned == 0.0, "offset predictions must not accidentally match truth"

    # And the fix: offsetting both sides equally is score-preserving.
    both_offset = entity_f_beta(
        {10 + offset, 20 + offset}, {t + offset for t in truth}
    )
    assert both_offset == 1.0
