"""Generate Kaggle kernel definitions for each pipeline stage.

Driving Kaggle through its API rather than the web UI matters here because the
run is five sequential stages of 1.5-4 hours each. Each stage's *output* is the
next stage's *input*, which the API expresses as `kernel_sources` -- so the
chain is declared once, in code, instead of being wired by hand five times with
a chance to get it wrong at 3am.

Kernels are `script` type, not notebooks: the logic lives in the repo's src/
and the kernel is a thin runner, which is what the graded reproducibility
requirement wants.

    python make_kernels.py --user zeroxbhuvii --dataset zeroxbhuvii/amazonml-er-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Stage -> the stages whose outputs it needs attached.
# train produces the model; each country needs the model; merge needs them all.
STAGES: dict[str, list[str]] = {
    "train":   [],
    "India":   ["train"],
    "US":      ["train"],
    "France":  ["train"],
    "merge":   ["train", "India", "US", "France"],
}


def slug(user: str, stage: str) -> str:
    return f"{user}/lassi-er-{stage.lower()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", required=True, help="Kaggle username")
    ap.add_argument("--dataset", required=True, help="owner/dataset-slug of the data")
    ap.add_argument("--out", default="kernels")
    ap.add_argument("--sample-entities", type=int, default=100_000)
    args = ap.parse_args()

    runner = Path(__file__).with_name("kaggle_run.py").read_text()
    root = Path(__file__).parent / args.out
    made = []

    for stage, depends_on in STAGES.items():
        d = root / stage.lower()
        d.mkdir(parents=True, exist_ok=True)

        # Bake the stage into the runner so the kernel needs no arguments.
        code = runner.replace('STAGE = "train"', f'STAGE = "{stage}"', 1)
        code = code.replace("SAMPLE_ENTITIES = 100_000",
                            f"SAMPLE_ENTITIES = {args.sample_entities}", 1)
        (d / "run.py").write_text(code)

        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": slug(args.user, stage),
            "title": f"lassi-er-{stage.lower()}",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,          # competition data must not go public
            "enable_gpu": False,         # CPU work; a GPU session gives FEWER vCPUs
            "enable_tpu": False,
            "enable_internet": True,     # pip install + git clone
            "dataset_sources": [args.dataset],
            "kernel_sources": [slug(args.user, s) for s in depends_on],
            "competition_sources": [],
        }, indent=2) + "\n")
        made.append((stage, d, depends_on))

    print(f"{'stage':<10}{'kernel id':<34}{'inherits from'}")
    print("-" * 74)
    for stage, d, deps in made:
        print(f"{stage:<10}{slug(args.user, stage):<34}{deps or '-'}")
    print(f"\nwrote {len(made)} kernel definitions under {root}")
    print("\nPush one with:")
    print(f"  kaggle kernels push -p {root}/train")
    print("Poll it with:")
    print(f"  kaggle kernels status {slug(args.user, 'train')}")


if __name__ == "__main__":
    main()
