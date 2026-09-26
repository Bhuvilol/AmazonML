"""Generate sharded Kaggle kernels for the recall-optimised rebuild.

Sharding exists because relaxing `max_df` from 0.01 to 0.5 recovers +2.6 points
of recall but makes blocking ~3x slower, which would put India at ~20 h against
Kaggle's 12 h session ceiling. Each shard processes a stride-slice of the SOURCE
entities against the FULL target pool, so recall is unaffected -- only
wall-clock time is divided.

Shard counts derive from the measured 597 s per 5,000 US entities at 6.19M
targets, scaled by each partition's entity and target counts, targeting ~4-5 h
per kernel.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

# (country, shards) -- from the measured time budget
PREDICT = [("India", 5), ("US", 3), ("France", 1)]
ARTIFACTS_DATASET = "lassi-er-artifacts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--artifacts-dataset", default=ARTIFACTS_DATASET)
    ap.add_argument("--sample-entities", type=int, default=30_000,
                    help="Training sample per country. Lower than before because "
                         "blocking is ~3x slower at max_df=0.5.")
    ap.add_argument("--commit", default="",
                    help="Pin kernels to this repo commit. Use whenever a run "
                         "must match an already-trained model: kernels clone at "
                         "runtime, so an unpinned kernel executes whatever is on "
                         "main when it STARTS, not when it was queued.")
    ap.add_argument("--out", default="kernels_v2")
    args = ap.parse_args()

    runner = Path(__file__).with_name("kaggle_run.py").read_text()
    if args.commit:
        before = runner
        runner = runner.replace('COMMIT = ""', f'COMMIT = "{args.commit}"', 1)
        assert runner != before, "COMMIT anchor not found in kaggle_run.py"
    root = Path(__file__).parent / args.out
    made = []

    def write(name: str, code: str, datasets: list[str]) -> None:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.py").write_text(code)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{args.user}/lassi2-{name}",
            "title": f"lassi2-{name}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": datasets,
            "kernel_sources": [],
            "competition_sources": [],
        }, indent=2) + "\n")
        made.append(name)

    # Training kernel
    code = runner.replace('STAGE = "train"', 'STAGE = "train"', 1)
    code = code.replace("SAMPLE_ENTITIES = 100_000",
                        f"SAMPLE_ENTITIES = {args.sample_entities}", 1)
    write("train", code, [args.dataset])

    # Sharded predict kernels
    for country, shards in PREDICT:
        for shard in range(shards):
            code = runner.replace('STAGE = "train"', f'STAGE = "{country}"', 1)
            # Inject shard flags into the predict command.
            code = code.replace(
                '"--country", STAGE,',
                f'"--country", STAGE, "--shard", "{shard}", "--shards", "{shards}",',
            )
            suffix = f"{country.lower()}-s{shard}of{shards}" if shards > 1 else country.lower()
            write(suffix, code, [args.dataset, f"{args.user}/{args.artifacts_dataset}"])

    print(f"{'kernel':<22}{'id'}")
    print("-" * 56)
    for name in made:
        print(f"{name:<22}{args.user}/lassi2-{name}")
    print(f"\n{len(made)} kernels in {root}")
    print(f"  1 train ({args.sample_entities:,} entities/country)")
    print(f"  {len(made)-1} predict shards: " +
          ", ".join(f"{c} x{s}" for c, s in PREDICT))


if __name__ == "__main__":
    main()
