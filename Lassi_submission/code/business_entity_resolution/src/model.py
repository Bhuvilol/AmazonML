"""Pairwise matching model: gradient-boosted trees over similarity features.

Why GBDT and not something bigger: the inputs are ~20 dense, bounded, tabular
similarity scores, which is precisely the regime where gradient boosting is
strongest. It trains in seconds on CPU (relevant -- the GPU quota for this
account is 0 and the increase request is still under human review), needs no
pretrained weights (so the MIT/Apache <=8B model rule is satisfied trivially
rather than argued), and produces feature importances that feed straight into
the required methodology write-up.

Two design points that matter more than the model choice:

**Sampling is by entity, never by pair.** The metric is macro-averaged per
Source-1 entity and the decision stage reasons over an entity's whole candidate
list, so training must see realistic per-entity candidate distributions. Drawing
a random sample of *pairs* would distort how many candidates each entity has and
break the link between training and the thing being optimised.

**Splits are grouped by entity.** A pair from entity X in train and another pair
from the same entity X in validation leaks: the two share a Source-1 record and
often near-duplicate Source-2/3 records. Grouping keeps the estimate honest.

Calibration matters here in a way it usually does not. The decision stage
compares P(match) against a threshold tuned on macro F_0.5, and later reasons
about "is this entity a singleton" from the candidate probabilities. That needs
probabilities, not just a ranking, so predictions are calibrated and checked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np

from features import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Deliberately modest. With ~20 informative features and millions of rows,
# capacity is not the binding constraint -- threshold placement is worth far
# more than tree depth, and every minute spent tuning these is a minute not
# spent on the decision stage.
DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": ["binary_logloss", "average_precision"],
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 0,       # 0 = all cores
    "verbosity": -1,
    "seed": 42,
    "deterministic": True,
}


@dataclass
class TrainedModel:
    """A fitted booster plus everything needed to reproduce and explain it."""

    booster: lgb.Booster
    feature_names: tuple[str, ...] = FEATURE_NAMES
    best_iteration: int = 0
    metrics: dict = field(default_factory=dict)

    def predict(self, features: np.ndarray, chunk_size: int = 2_000_000) -> np.ndarray:
        """Predict match probabilities, chunked to bound peak memory."""
        if len(features) <= chunk_size:
            return self.booster.predict(
                features, num_iteration=self.best_iteration or None
            ).astype(np.float32)

        out = np.empty(len(features), dtype=np.float32)
        for start in range(0, len(features), chunk_size):
            stop = min(start + chunk_size, len(features))
            out[start:stop] = self.booster.predict(
                features[start:stop], num_iteration=self.best_iteration or None
            )
        return out

    def importance_report(self, top: int = 20) -> list[tuple[str, float]]:
        """Feature importances by gain, for the methodology document."""
        gains = self.booster.feature_importance(importance_type="gain")
        paired = sorted(zip(self.feature_names, gains), key=lambda kv: -kv[1])
        total = sum(gains) or 1.0
        return [(name, 100.0 * gain / total) for name, gain in paired[:top]]

    def save(self, path) -> None:
        self.booster.save_model(str(path), num_iteration=self.best_iteration or None)


def entity_group_split(
    entity_index: np.ndarray,
    n_entities: int,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split candidate pairs into train/valid by *entity*, never by pair.

    Args:
        entity_index: source-entity index for each candidate pair.
        n_entities: total number of source entities.
        valid_fraction: share of entities held out.
        seed: RNG seed, fixed for reproducibility.

    Returns:
        Boolean masks ``(train_mask, valid_mask)`` over the pair arrays.
    """
    rng = np.random.default_rng(seed)
    held_out = np.zeros(n_entities, dtype=bool)
    chosen = rng.choice(
        n_entities, size=max(1, int(n_entities * valid_fraction)), replace=False
    )
    held_out[chosen] = True

    valid_mask = held_out[entity_index]
    return ~valid_mask, valid_mask


def train(
    features: np.ndarray,
    labels: np.ndarray,
    entity_index: np.ndarray,
    n_entities: int,
    params: dict | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
    valid_fraction: float = 0.2,
    seed: int = 42,
) -> TrainedModel:
    """Fit the pairwise matcher with an entity-grouped validation split."""
    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

    train_mask, valid_mask = entity_group_split(
        entity_index, n_entities, valid_fraction, seed
    )
    logger.info(
        "train %d pairs (%.2f%% positive) | valid %d pairs (%.2f%% positive)",
        train_mask.sum(), 100 * labels[train_mask].mean(),
        valid_mask.sum(), 100 * labels[valid_mask].mean(),
    )

    train_set = lgb.Dataset(
        features[train_mask], label=labels[train_mask],
        feature_name=list(FEATURE_NAMES), free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        features[valid_mask], label=labels[valid_mask],
        reference=train_set, feature_name=list(FEATURE_NAMES), free_raw_data=False,
    )

    evals: dict = {}
    booster = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set], valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals),
            lgb.log_evaluation(period=100),
        ],
    )

    model = TrainedModel(
        booster=booster,
        best_iteration=booster.best_iteration or num_boost_round,
        metrics={k: v[-1] for k, v in evals.get("valid", {}).items()},
    )
    logger.info("best_iteration=%d metrics=%s", model.best_iteration, model.metrics)
    return model


def calibration_report(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> str:
    """Compare predicted probability against observed rate, per decile.

    The decision stage treats these values as probabilities -- it compares them
    to a threshold and uses them to judge whether an entity has any match at
    all. If they are badly calibrated, a threshold tuned on validation will sit
    in the wrong place on test. This check is cheap insurance.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    lines = [f"{'bin':>12}{'n':>10}{'pred':>9}{'actual':>9}{'gap':>8}"]
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probabilities >= lo) & (probabilities < hi if i < bins - 1 else probabilities <= hi)
        if not mask.any():
            continue
        pred, actual = probabilities[mask].mean(), labels[mask].mean()
        lines.append(
            f"{f'[{lo:.1f},{hi:.1f})':>12}{mask.sum():>10,}"
            f"{pred:>9.3f}{actual:>9.3f}{pred - actual:>+8.3f}"
        )
    return "\n".join(lines)
