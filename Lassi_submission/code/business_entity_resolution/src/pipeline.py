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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import blocking as B
import decide as DEC
import model as M
from features import RecordArrays, compute_pair_features
from io_tsv import _READ_OPTS, read_entity_ids
from submit import SEP as SEP_TAB
from submit import CANDIDATE_HEADER, MATCHING_HEADER, StreamingIdListWriter

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]

# Paths are environment-overridable so the same code runs unchanged locally and
# on a hosted notebook (Kaggle mounts read-only inputs under /kaggle/input and
# only /kaggle/working is writable). Hard-coded paths would have forced a fork
# of the pipeline, which the graded reproducibility requirement disallows.
DATA_ROOT = Path(
    os.environ.get("LASSI_DATA_ROOT", REPO_ROOT / "student_resource" / "dataset")
)
ARTIFACTS = Path(
    os.environ.get("LASSI_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts")
)
OUTPUT_DIR = Path(
    os.environ.get("LASSI_OUTPUT", REPO_ROOT / "Lassi_submission" / "output")
)

# Worker count. sparse_dot_topn treats 0/None as SERIAL, so -1 (all cores) is
# the correct default. On a small hosted box leaving one core free avoids
# starving the OS, which is what caused a local machine reset under full load.
def default_threads() -> int:
    override = os.environ.get("LASSI_THREADS")
    if override:
        return int(override)
    cores = os.cpu_count() or 1
    return -1 if cores > 4 else max(1, cores - 1)


@dataclass
class BlockingConfig:
    """Tuned on training data; see CONTEXT.md for the sweeps behind each value."""

    # Swept against the full 6.19M target pool: recall 0.9279 @20, 0.9397 @30,
    # 0.9462 @40. Blocking time is FLAT across top_n (~305s either way), so the
    # only cost of a larger k is downstream featurisation.
    top_n_name: int = 40
    top_n_addr: int = 40
    # Measured inert: 0.25 vs 0.10 gave identical recall (0.9462 both). The
    # similarity floor never binds -- top_n is the sole limiter. Kept as a
    # cheap guard against pathological low-similarity candidates.
    min_sim_name: float = 0.25
    min_sim_addr: float = 0.30
    # max_df is the dominant speed knob: 0.01 gives a 21x speedup over no
    # pruning for ~1.7 points of recall. 0.003 collapses recall to 0.84.
    max_df: float = 0.01
    ngram_range: tuple[int, int] = (3, 3)
    # MUST NOT be 0/None: sparse_dot_topn treats those as serial
    # (`n_threads or 1`), which silently costs ~2.7x.
    n_threads: int = field(default_factory=default_threads)


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


def entity_ids_for_country(
    split: str, country: str, shard: int = 0, shards: int = 1
) -> list[str]:
    """Source-1 entity ids for one country, optionally one shard of them."""
    path = DATA_ROOT / split / f"{split}_source1.tsv"
    ids = (
        pl.scan_csv(path, **_READ_OPTS)
        .filter(pl.col("country") == country)
        .select("entity_id").collect()["entity_id"].to_list()
    )
    return ids[shard::shards] if shards > 1 else ids


def run_predict(
    split: str,
    trained: M.TrainedModel,
    threshold: float,
    config: BlockingConfig,
    output_dir: Path,
    singleton_gate: float | None = None,
    enforce_exclusivity: bool = True,
    countries: list[str] | None = None,
    suffix: str = "",
    threshold_report: bool = False,
    shard: int = 0,
    shards: int = 1,
) -> None:
    """Generate both submission files for a split.

    Rows are streamed per country partition rather than accumulated. At 1.73M
    entities and ~77 candidates each, holding the full mapping first would cost
    several GB on top of the live sparse matrices -- and an OOM at the final
    write would destroy a multi-hour run. Streaming is safe because the official
    validator compares required/seen as sets and never checks row order.
    """
    targets_countries = countries or discover_countries(split)
    partial = countries is not None

    if partial:
        # A partial run is complete with respect to its own countries (and
        # shard) only.
        required = [
            e for c in targets_countries
            for e in entity_ids_for_country(split, c, shard, shards)
        ]
        logger.info(
            "PARTIAL run for %s shard %d/%d: %d entities",
            targets_countries, shard, shards, len(required),
        )
    else:
        required = read_entity_ids(DATA_ROOT / split / f"{split}_source1.tsv")

    output_dir.mkdir(parents=True, exist_ok=True)

    matching = StreamingIdListWriter(
        output_dir / f"matching_results{suffix}.tsv", MATCHING_HEADER, required
    )
    candidates_file = StreamingIdListWriter(
        output_dir / f"candidate_pairs{suffix}.tsv", CANDIDATE_HEADER, required
    )

    with matching, candidates_file:
        for country in targets_countries:
            started = time.time()
            source1 = load_partition(split, 1, country)
            if shards > 1:
                # Stride-slice the SOURCE side only. Each shard still sees the
                # full target pool, so recall is unaffected -- this splits
                # wall-clock time, not the candidate space.
                source1 = source1.with_row_index("_row").filter(
                    pl.col("_row") % shards == shard
                ).drop("_row")
            targets = pl.concat([load_partition(split, n, country) for n in (2, 3)])
            logger.info(
                "[%s] %s shard %d/%d: %d source x %d targets",
                split, country, shard, shards, source1.height, targets.height,
            )

            candidates, stats = build_partition_candidates(source1, targets, config)
            logger.info("[%s] %s blocking: %s", split, country, stats.describe())

            probabilities = score_candidates(source1, targets, candidates, trained)

            if threshold_report:
                # One run yields the whole threshold curve instead of one point.
                # Needed because France cannot be validated locally -- there is
                # no French training data -- so the only way to choose its
                # threshold is to see how the abstention rate responds and
                # compare against countries that ARE in training.
                logger.info("[%s] threshold report (train singleton rate 5.58%%):", country)
                logger.info("    %7s %10s %8s %12s", "thresh", "empty%", "mean", "total_ids")
                for probe in [round(x, 3) for x in np.arange(0.20, 0.86, 0.05)]:
                    picks = DEC.select(
                        candidates.row, candidates.col, probabilities,
                        source1.height, threshold=float(probe),
                        singleton_gate=None, enforce_exclusivity=enforce_exclusivity,
                    )
                    sizes = np.fromiter((len(p) for p in picks), dtype=np.int32,
                                        count=source1.height)
                    empty = int((sizes == 0).sum())
                    nonzero = sizes[sizes > 0]
                    logger.info(
                        "    %7.3f %9.2f%% %8.2f %12d", probe,
                        100 * empty / source1.height,
                        float(nonzero.mean()) if len(nonzero) else 0.0,
                        int(sizes.sum()),
                    )

            selected = DEC.select(
                candidates.row, candidates.col, probabilities, source1.height,
                threshold=threshold, singleton_gate=singleton_gate,
                enforce_exclusivity=enforce_exclusivity,
            )

            s1_ids = source1["entity_id"].to_list()
            tgt_ids = targets["entity_id"].to_list()
            for row in range(source1.height):
                entity_id = s1_ids[row]
                span = candidates.row_slice(row)
                candidates_file.write_row(
                    entity_id, (tgt_ids[c] for c in candidates.col[span])
                )
                matching.write_row(entity_id, (tgt_ids[c] for c in selected[row]))

            emitted = sum(1 for chosen in selected if len(chosen))
            logger.info(
                "[%s] %s done in %.0fs | %d/%d entities matched (%.1f%%)",
                split, country, time.time() - started,
                emitted, source1.height, 100 * emitted / source1.height,
            )
            del source1, targets, candidates, probabilities, selected, s1_ids, tgt_ids


def merge_partials(split: str, output_dir: Path) -> None:
    """Concatenate per-country partial outputs into the final submission files.

    Verifies that the union of partials covers every Source-1 test entity
    exactly once before writing, so a missing or duplicated country partition
    fails here rather than at submission.
    """
    required = read_entity_ids(DATA_ROOT / split / f"{split}_source1.tsv")

    for stem, header in (("matching_results", MATCHING_HEADER),
                         ("candidate_pairs", CANDIDATE_HEADER)):
        partials = sorted(output_dir.glob(f"{stem}_*.tsv"))
        if not partials:
            raise SystemExit(f"No partial files matching {stem}_*.tsv in {output_dir}")
        logger.info("merging %d partial(s) into %s.tsv: %s",
                    len(partials), stem, [p.name for p in partials])

        seen: set[str] = set()
        destination = output_dir / f"{stem}.tsv"
        with destination.open("w", encoding="utf-8", newline="") as out:
            out.write(SEP_TAB.join(header) + "\n")
            for part in partials:
                with part.open(encoding="utf-8") as handle:
                    handle.readline()   # skip the partial's header
                    for line in handle:
                        entity_id = line.split(SEP_TAB, 1)[0]
                        if entity_id in seen:
                            raise SystemExit(
                                f"{part.name}: {entity_id} already written by an "
                                f"earlier partial -- overlapping country partitions."
                            )
                        seen.add(entity_id)
                        out.write(line)

        missing = set(required) - seen
        if missing:
            raise SystemExit(
                f"{stem}.tsv is incomplete: {len(missing)} entities missing "
                f"(e.g. {sorted(missing)[:5]}). Run the remaining country partitions."
            )
        logger.info("wrote %s (%d rows)", destination, len(seen))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict", "merge"])
    parser.add_argument(
        "--country", action="append", default=None,
        help="Restrict to one country partition (repeatable). Produces partial "
             "output files suffixed with the country name, so a long run can be "
             "split across sessions and merged afterwards. Countries are "
             "independent: true matches never cross them.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default=str(ARTIFACTS / "model.txt"))
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold. Defaults to the value train_model.py selected "
             "in artifacts/threshold.json, so the two stages cannot drift apart.",
    )
    parser.add_argument("--singleton-gate", type=float, default=None)
    parser.add_argument("--no-exclusivity", action="store_true")
    parser.add_argument(
        "--threshold-report", action="store_true",
        help="Also print abstention rate and mean match count across a "
             "threshold grid. One run then answers 'what threshold gives "
             "sensible behaviour here', instead of one run per candidate value.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--sample-entities", type=int, default=150_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "predict":
        import json

        import lightgbm as lgb

        threshold = args.threshold
        if threshold is None:
            threshold_path = Path(args.model).parent / "threshold.json"
            if not threshold_path.is_file():
                raise SystemExit(
                    f"No --threshold given and {threshold_path} not found. "
                    f"Run train_model.py first, or pass --threshold explicitly."
                )
            payload = json.loads(threshold_path.read_text())
            threshold = float(payload["threshold"])
            logger.info(
                "threshold %.3f from %s (validation macro F0.5 %.4f)",
                threshold, threshold_path.name, payload.get("valid_macro_f05", float("nan")),
            )

        booster = lgb.Booster(model_file=args.model)
        trained = M.TrainedModel(booster=booster, best_iteration=booster.num_trees())
        suffix = f"_{'_'.join(args.country)}" if args.country else ""
        if args.shards > 1:
            suffix += f"_s{args.shard}of{args.shards}"
        run_predict(
            args.split, trained, threshold, BlockingConfig(),
            Path(args.output_dir), args.singleton_gate, not args.no_exclusivity,
            countries=args.country, suffix=suffix,
            threshold_report=args.threshold_report,
            shard=args.shard, shards=args.shards,
        )
        return 0

    if args.command == "merge":
        merge_partials(args.split, Path(args.output_dir))
        return 0

    raise SystemExit("train is driven by train_model.py; see README")


if __name__ == "__main__":
    raise SystemExit(main())
