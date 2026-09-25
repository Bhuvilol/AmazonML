# ==========================================================================
# ML Challenge 2026 - Business Entity Resolution : Kaggle runner
#
# Paste this into ONE Kaggle notebook cell. Change STAGE between runs.
#
# Requires, in notebook Settings:
#   * Internet: ON        (pip install + git clone)
#   * Accelerator: None   (this is CPU work; a GPU session gives FEWER vCPUs)
#   * The dataset attached (see KAGGLE_SETUP.md)
#
# Why split by stage: at ~4 vCPU the full test pass runs ~8h against Kaggle's
# 12h ceiling. Countries are independent partitions (true matches never cross
# them, verified on 693k training pairs), so each runs separately and a failure
# costs one partition instead of the whole job.
# ==========================================================================

STAGE = "France"          # "train" | "India" | "US" | "France" | "merge"
SAMPLE_ENTITIES = 100000
REPO = "https://github.com/Bhuvilol/AmazonML.git"

# --------------------------------------------------------------------------
import os, subprocess, sys, time, glob, shutil, textwrap
from pathlib import Path

t_start = time.time()

def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)

# ---- 1. Report the ACTUAL machine. Never assume Kaggle's spec. -----------
print("=" * 62)
sh("nproc; free -g | head -2; df -h /kaggle/working | tail -1", check=False)
print("cpu_count:", os.cpu_count())
print("=" * 62, flush=True)

# ---- 2. Dependencies -----------------------------------------------------
# Most are preinstalled on Kaggle; these two usually are not.
sh("pip install -q sparse_dot_topn rapidfuzz polars 2>&1 | tail -2", check=False)

# ---- 3. Code: cloned from the repo so the notebook stays a THIN RUNNER ---
# Reproducibility is graded: all logic must live in src/, never in cells.
# Cloned to /tmp, NOT /kaggle/working: everything under working becomes the
# kernel's output and is re-mounted into downstream stages. Putting the repo
# there would bloat every stage's output with a copy of the source tree.
CHECKOUT = Path("/tmp/AmazonML")
SRC = CHECKOUT / "Lassi_submission/code/business_entity_resolution/src"
if not SRC.exists():
    sh(f"git clone --depth 1 {REPO} {CHECKOUT}")
assert SRC.exists(), f"src not found at {SRC}"
sh(f"cd {CHECKOUT} && git log -1 --oneline", check=False)   # pin what ran

# ---- 4. Locate the attached dataset --------------------------------------
def find_data_root() -> Path:
    """Locate the dataset wherever it ended up under /kaggle/input.

    Searched recursively rather than at fixed depths: Kaggle nests uploads
    differently depending on whether you upload a folder, a zip, or individual
    files, and guessing wrong would fail after the session had already started.
    We anchor on train_source1.tsv and take its grandparent as the root.
    """
    hits = sorted(Path("/kaggle/input").rglob("train_source1.tsv"))
    if hits:
        root = hits[0].parent.parent          # .../<root>/train/train_source1.tsv
        if (root / "test" / "test_source1.tsv").exists():
            return root
        print(f"WARNING: found {hits[0]} but no test/test_source1.tsv beside it")
        return root
    listing = [str(p) for p in Path("/kaggle/input").rglob("*")][:40]
    raise SystemExit(
        "Could not find train_source1.tsv anywhere under /kaggle/input.\n"
        "Is the dataset attached? (Notebook -> Add Input -> your dataset)\n"
        "Contents seen:\n  " + "\n  ".join(listing)
    )

DATA_ROOT = find_data_root()
print("DATA_ROOT:", DATA_ROOT, flush=True)
sh(f"ls -la {DATA_ROOT}/train {DATA_ROOT}/test", check=False)

# ---- 5. Environment ------------------------------------------------------
ARTIFACTS = Path("/kaggle/working/artifacts")
OUTPUT    = Path("/kaggle/working/output")
ARTIFACTS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# A previous stage's artifacts arrive as another attached dataset -- copy them
# in so this stage can read them.
for prior in glob.glob("/kaggle/input/*/artifacts/model.txt"):
    shutil.copytree(Path(prior).parent, ARTIFACTS, dirs_exist_ok=True)
    print("reused artifacts from", prior, flush=True)
for prior in glob.glob("/kaggle/input/*/output/*_*.tsv"):
    shutil.copy(prior, OUTPUT / Path(prior).name)
    print("reused partial output", Path(prior).name, flush=True)

env = dict(os.environ)
env["LASSI_DATA_ROOT"] = str(DATA_ROOT)
env["LASSI_ARTIFACTS"] = str(ARTIFACTS)
env["LASSI_OUTPUT"]    = str(OUTPUT)
env["PYTHONUNBUFFERED"] = "1"

# ---- 6. Run --------------------------------------------------------------
if STAGE == "train":
    cmd = [sys.executable, "train_model.py",
           "--sample-entities", str(SAMPLE_ENTITIES),
           "--artifacts", str(ARTIFACTS)]
elif STAGE == "merge":
    cmd = [sys.executable, "-m", "pipeline", "merge",
           "--output-dir", str(OUTPUT)]
else:
    cmd = [sys.executable, "-m", "pipeline", "predict",
           "--model", str(ARTIFACTS / "model.txt"),
           "--country", STAGE,
           "--output-dir", str(OUTPUT)]

print("\n" + "=" * 62)
print("STAGE:", STAGE)
print("CMD  :", " ".join(cmd))
print("=" * 62, flush=True)

proc = subprocess.run(cmd, cwd=str(SRC), env=env)
print(f"\nexit={proc.returncode}   elapsed={(time.time()-t_start)/60:.1f} min", flush=True)

# ---- 7. Report what was produced ----------------------------------------
sh(f"ls -lh {ARTIFACTS} {OUTPUT} 2>/dev/null", check=False)

# Outputs are large (candidate_pairs.tsv is ~1.9 GB uncompressed); gzip so the
# notebook output stays small enough to download comfortably.
if STAGE == "merge":
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        f = OUTPUT / name
        if f.exists():
            sh(f"gzip -kf {f}", check=False)
    sh(f"ls -lh {OUTPUT}/*.gz", check=False)

if proc.returncode != 0:
    raise SystemExit(f"stage {STAGE} failed with exit code {proc.returncode}")
