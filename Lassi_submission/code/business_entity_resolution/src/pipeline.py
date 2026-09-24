"""End-to-end pipeline: data -> blocking -> matching -> output.

One country partition at a time. That is not an optimisation detail -- it is
correctness plus memory. Matches never cross countries (verified: 693,069
training pairs, zero cross-country), so partitioning is lossless, and it keeps
peak memory inside 8 GB by never holding more than one partition's sparse
matrices at once.

Country is discovered from the data, never hard-coded. The test set contains
France, which is absent from training, and the problem statement explicitly
forbids restricting the pipeline to the training countries. Nothing here
branches on a country's name; an unseen label simply becomes another partition.

Entry points:
    python -m pipeline train    --sample-entities 150000
    python -m pipeline predict  --model artifacts/model.txt
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import blocking as B
import decide as DEC
import model as M
from features import RecordArrays, compute_pair_features
from io_tsv import _READ_OPTS, read_entity_ids
from submit import write_candidate_pairs, write_matching_results

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "student_resource" / "dataset"
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


@dataclass
class BlockingConfig:
    """Tuned on training data; see CONTEXT.md for the sweeps behind each value."""

    top_n_name: int = 20
    top_n_addr: int = 20
    min_sim_name: float = 0.25
    min_sim_addr: float = 0.30
    # max_df is the dominant speed knob: 0.01 gives a 21x speedup over no
    # pruning for ~1.7 points of recall. 0.003 collapses recall to 0.84.
    max_df: float = 0.01
    ngram_range: tuple[int, int] = (3, 3)
    # MUST be -1. sparse_dot_topn treats 0/None as serial (`n_threads or 1`).
    n_threads: int = -1


def load_partition(split: str, source: int, country: str | None = None) -> pl.DataFrame:
    """Load one source file, optionally restricted to a country."""
    from normalize import add_normalized_columns

    path = DATA_ROOT / split / f"{split}_source{source}.tsv"
    frame = pl.scan_csv(path, **_READ_OPTS)
    if country is not None:
        frame = frame.filter(pl.col("country") == country)
    return add_normalized_columns(frame).collect()


def discover_countries(split: str) -> list[str]:
    """Read the country labels present in a split's Source-1 file.

    Discovered, never hard-coded -- the test set adds France, and the problem
    statement forbids a pipeline restricted to {US, India}.
    """
    path = DATA_ROOT / split / f"{split}_source1.tsv"
    values = (
        pl.scan_csv(path, **_READ_OPTS)
        .select("country").unique().collect()["country"].to_list()
    )
    countries = sorted(v for v in values if v)
    logger.info("discovered countries in %s: %s", split, countries)
    return countries


def build_partition_candidates(
    source1: pl.DataFrame, targets: pl.DataFrame, config: BlockingConfig
) -> tuple[B.CandidateSet, B.BlockingStats]:
    return B.build_candidates(
        source1["name_block"].to_list(), source1["addr_block"].to_list(),
        targets["name_block"].to_list(), targets["addr_block"].to_list(),
        top_n_name=config.top_n_name, top_n_addr=config.top_n_addr,
        min_sim_name=config.min_sim_name, min_sim_addr=config.min_sim_addr,
        max_df=config.max_df, ngram_range=config.ngram_range,
        n_threads=config.n_threads,
    )


def score_candidates(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
    candidates: B.CandidateSet,
    trained: M.TrainedModel,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """Featurise and score every candidate pair, chunked.

    Chunking is required, not optional: the full test candidate set is ~65M
    pairs, and a single float32 feature matrix for that is ~5 GB against 8 GB
    of RAM.
    """
    left = RecordArrays.from_frame(source1)
    right = RecordArrays.from_frame(targets)
    out = np.empty(len(candidates), dtype=np.float32)

    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        features = compute_pair_features(
            left, right,
            candidates.row[start:stop], candidates.col[start:stop],
            candidates.name_cos[start:stop], candidates.addr_cos[start:stop],
        )
        out[start:stop] = trained.predict(features)
        logger.info("  scored %d/%d pairs", stop, len(candidates))
        del features
    return out


def truth_for_partition(
    split: str, entity_ids: list[str], target_index: dict[str, int]
) -> list[set[int]]:
    """Ground-truth target indices per Source-1 entity, as integer sets."""
    index = {e: i for i, e in enumerate(entity_ids)}
    truth: list[set[int]] = [set() for _ in entity_ids]
    path = DATA_ROOT / split / f"{split}_ground_truth.tsv"

    frame = (
        pl.scan_csv(path, **_READ_OPTS)
        .filter(pl.col("source1_entity_id").is_in(pl.Series(entity_ids).implode()))
        .collect()
    )
    for entity_id, matched in zip(frame["source1_entity_id"], frame["matched_entity_ids"]):
        if not matched:
            continue
        row = index[entity_id]
        for mid in matched.split(","):
            position = target_index.get(mid)
            if position is not None:
                truth[row].add(position)
    return truth


def run_predict(
    split: str,
    trained: M.TrainedModel,
    threshold: float,
    config: BlockingConfig,
    output_dir: Path,
    singleton_gate: float | None = None,
    enforce_exclusivity: bool = True,
) -> None:
    """Generate both submission files for a split."""
    required = read_entity_ids(DATA_ROOT / split / f"{split}_source1.tsv")
    matches: dict[str, list[str]] = {}
    candidates_out: dict[str, list[str]] = {}

    for country in discover_countries(split):
        started = time.time()
        source1 = load_partition(split, 1, country)
        targets = pl.concat([load_partition(split, n, country) for n in (2, 3)])
        logger.info(
            "[%s] %s: %d source x %d targets",
            split, country, source1.height, targets.height,
        )

        candidates, stats = build_partition_candidates(source1, targets, config)
        logger.info("[%s] %s blocking: %s", split, country, stats.describe())

        probabilities = score_candidates(source1, targets, candidates, trained)
        selected = DEC.select(
            candidates.row, candidates.col, probabilities, source1.height,
            threshold=threshold, singleton_gate=singleton_gate,
            enforce_exclusivity=enforce_exclusivity,
        )

        s1_ids = source1["entity_id"].to_list()
        tgt_ids = targets["entity_id"].to_list()
        for row, chosen in enumerate(selected):
            matches[s1_ids[row]] = [tgt_ids[c] for c in chosen]
        for row in range(source1.height):
            span = candidates.row_slice(row)
            candidates_out[s1_ids[row]] = [tgt_ids[c] for c in candidates.col[span]]

        logger.info("[%s] %s done in %.0fs", split, country, time.time() - started)
        del source1, targets, candidates, probabilities

    output_dir.mkdir(parents=True, exist_ok=True)
    write_matching_results(output_dir / "matching_results.tsv", matches, required)
    write_candidate_pairs(output_dir / "candidate_pairs.tsv", candidates_out, required)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=str(ARTIFACTS / "model.txt"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--singleton-gate", type=float, default=None)
    parser.add_argument("--no-exclusivity", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "Lassi_submission" / "output"))
    parser.add_argument("--sample-entities", type=int, default=150_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "predict":
        import lightgbm as lgb

        booster = lgb.Booster(model_file=args.model)
        trained = M.TrainedModel(booster=booster, best_iteration=booster.num_trees())
        run_predict(
            args.split, trained, args.threshold, BlockingConfig(),
            Path(args.output_dir), args.singleton_gate, not args.no_exclusivity,
        )
        return 0

    raise SystemExit("train is driven by train_model.py; see README")


if __name__ == "__main__":
    raise SystemExit(main())
