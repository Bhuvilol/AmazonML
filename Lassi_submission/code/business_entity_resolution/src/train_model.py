"""Train the pairwise matcher and select the decision threshold.

Writes ``artifacts/model.txt`` and ``artifacts/threshold.json``, which
``pipeline.py predict`` consumes. Run from this directory:

    python train_model.py --sample-entities 150000

Two sampling decisions matter and are deliberate:

**Sample Source-1 entities, never candidate pairs.** The metric is macro
averaged per entity and the decision stage reasons over an entity's whole
candidate list. Sampling pairs would distort per-entity candidate counts and
sever the link between what is trained and what is scored.

**Keep the full target pool for every sampled entity.** An earlier version
sampled distractors, which made the problem look far easier than it is
(recall ceiling 0.97 against a 369k pool versus 0.93 against the real 6.2M
pool) and introduced a spurious S2/S3 signal, because the truncated pool was
almost entirely Source 2. Negatives must come from the real competition.

The threshold is tuned on a held-out set of entities against macro F_0.5
directly -- never on the public leaderboard, whose score is a noisy macro
average over a subset while final rankings come from the private split.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

import blocking as B
import decide as DEC
import model as M
from features import RecordArrays, compute_pair_features
from io_tsv import _READ_OPTS
from pipeline import ARTIFACTS, DATA_ROOT, BlockingConfig, load_partition, discover_countries

logger = logging.getLogger(__name__)


def build_training_partition(country: str, n_entities: int, seed: int, config: BlockingConfig):
    """Block one country's sampled entities against its FULL target pool."""
    source1 = load_partition("train", 1, country)
    if n_entities < source1.height:
        source1 = source1.sample(n=n_entities, seed=seed)
    targets = pl.concat([load_partition("train", n, country) for n in (2, 3)])
    logger.info("[%s] %d sampled source x %d targets", country, source1.height, targets.height)

    candidates, stats = B.build_candidates(
        source1["name_block"].to_list(), source1["addr_block"].to_list(),
        targets["name_block"].to_list(), targets["addr_block"].to_list(),
        top_n_name=config.top_n_name, top_n_addr=config.top_n_addr,
        min_sim_name=config.min_sim_name, min_sim_addr=config.min_sim_addr,
        max_df=config.max_df, ngram_range=config.ngram_range,
        n_threads=config.n_threads,
    )
    logger.info("[%s] %s", country, stats.describe())

    # Ground truth as integer target indices.
    target_index = {e: i for i, e in enumerate(targets["entity_id"].to_list())}
    entity_ids = source1["entity_id"].to_list()
    entity_index = {e: i for i, e in enumerate(entity_ids)}
    truth: list[set[int]] = [set() for _ in entity_ids]

    gt = (
        pl.scan_csv(DATA_ROOT / "train" / "train_ground_truth.tsv", **_READ_OPTS)
        .filter(pl.col("source1_entity_id").is_in(pl.Series(entity_ids).implode()))
        .collect()
    )
    n_true = 0
    for sid, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if not matched:
            continue
        row = entity_index[sid]
        for mid in matched.split(","):
            position = target_index.get(mid)
            if position is not None:
                truth[row].add(position)
                n_true += 1

    positive_keys = {(r, c) for r, s in enumerate(truth) for c in s}
    labels = np.fromiter(
        ((r, c) in positive_keys for r, c in zip(candidates.row.tolist(), candidates.col.tolist())),
        dtype=np.int8, count=len(candidates),
    )
    recall = labels.sum() / n_true if n_true else float("nan")
    logger.info(
        "[%s] recall ceiling %.4f | %d positives of %d candidates (%.2f%%)",
        country, recall, labels.sum(), len(candidates), 100 * labels.mean(),
    )

    left = RecordArrays.from_frame(source1)
    right = RecordArrays.from_frame(targets)
    features = compute_pair_features(
        left, right, candidates.row, candidates.col,
        candidates.name_cos, candidates.addr_cos,
    )
    return features, labels, candidates, truth, source1.height, targets.height, recall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-entities", type=int, default=150_000,
                        help="Source-1 entities sampled per country")
    parser.add_argument("--valid-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifacts", default=str(ARTIFACTS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    config = BlockingConfig()
    started = time.time()

    all_features, all_labels, all_rows, all_cols, all_truth = [], [], [], [], []
    recalls: dict[str, float] = {}
    entity_offset = 0
    target_offset = 0
    for country in discover_countries("train"):
        features, labels, candidates, truth, n_entities, n_targets, recall = (
            build_training_partition(country, args.sample_entities, args.seed, config)
        )
        all_features.append(features)
        all_labels.append(labels)
        # Offset entity indices so every partition shares one global numbering.
        all_rows.append(candidates.row.astype(np.int64) + entity_offset)
        # Target indices are partition-local. They must be made globally unique
        # so exclusivity resolution never merges two different countries'
        # records that happen to share a local index -- and CRITICALLY the same
        # offset must be applied to the ground truth, or predictions and truth
        # live in different index spaces and every intersection is empty.
        all_cols.append(candidates.col.astype(np.int64) + target_offset)
        all_truth.extend({c + target_offset for c in entity_truth} for entity_truth in truth)
        recalls[country] = recall
        entity_offset += n_entities
        target_offset += n_targets

    features = np.vstack(all_features); del all_features
    labels = np.concatenate(all_labels).astype(np.float64)
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    logger.info("combined: %s features, %d entities", features.shape, entity_offset)

    trained = M.train(features, labels, rows, entity_offset,
                      valid_fraction=args.valid_fraction, seed=args.seed)
    print("\nFEATURE IMPORTANCE (gain %):")
    for name, gain in trained.importance_report(20):
        print(f"  {name:<24}{gain:>6.1f}")

    # ---- Threshold selection on the SAME held-out entities the model never saw.
    # Tuned against macro F_0.5 directly, never against the public leaderboard:
    # final rankings come from the private split and the public score is a noisy
    # macro average over a subset.
    _, valid_mask = M.entity_group_split(rows, entity_offset, args.valid_fraction, args.seed)
    probabilities = trained.predict(features)

    valid_entities = np.unique(rows[valid_mask])
    remap = {e: i for i, e in enumerate(valid_entities)}
    local_rows = np.fromiter((remap[r] for r in rows[valid_mask]),
                             dtype=np.int64, count=int(valid_mask.sum()))
    valid_truth = [all_truth[e] for e in valid_entities]

    threshold, score, curve = DEC.sweep_threshold(
        local_rows, cols[valid_mask], probabilities[valid_mask], valid_truth
    )
    empty_baseline = sum(1 for t in valid_truth if not t) / len(valid_truth)
    print(f"\nBEST THRESHOLD {threshold:.3f} -> macro F0.5 {score:.4f} "
          f"(all-empty baseline {empty_baseline:.4f})")

    trained.save(artifacts / "model.txt")
    (artifacts / "threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "valid_macro_f05": score,
        "all_empty_baseline": empty_baseline,
        "curve": curve,
    }, indent=2))
    (artifacts / "blocking_recall.json").write_text(json.dumps(recalls, indent=2))
    logger.info("saved model + threshold to %s (%.0fs total)", artifacts, time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
