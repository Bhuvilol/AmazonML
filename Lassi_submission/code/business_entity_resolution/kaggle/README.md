# Kaggle execution harness

These are the exact notebooks that produced the submitted `output/` files. They
are **thin runners only** — no pipeline logic lives here. Every runner clones
this repository at a pinned commit and calls `src/`, so reproducibility does
not depend on any notebook state.

## Files

| Path | Purpose |
|---|---|
| `kaggle_run.py` | The runner template. Reports real machine specs, installs deps, clones the repo **at a pinned commit**, locates the attached dataset, and invokes `src/train_model.py` or `src/pipeline.py`. |
| `make_kernels.py` | Generates the ten kernel folders below from the template. |
| `kernels/<name>/run.py` | Generated runner, one per stage. |
| `kernels/<name>/kernel-metadata.json` | Kaggle kernel definition (datasets attached, internet on, CPU). |

## The run that produced the submission

All ten kernels are pinned to commit **`18134aa`** — the frozen production
configuration. `COMMIT` is asserted at runtime via `git rev-parse`, so a
silently-failed checkout cannot reintroduce a mismatch.

| Stage | Kernel | Runtime |
|---|---|---|
| Training | `train` | 2 h 26 m |
| India, 5 shards | `india-s0of5` … `india-s4of5` | ~6 h 30 m |
| US, 3 shards | `us-s0of3` … `us-s2of3` | ~5 h |
| France, 1 shard | `france` | ~2 h 15 m |

Sharding is by Source-1 entity range (`--shard N --shards M`); each shard
blocks its slice against the **full** country target pool, so shard count does
not affect results — only wall-clock time. `src/pipeline.py merge` reassembles
the partials and fails loudly if any test entity is missing.

## Why pinning matters

An earlier run used `git clone --depth 1`, which resolves to whatever is on
`main` **when the kernel starts**, not when it was queued. A feature change
pushed while shards were waiting for compute slots caused one shard to load a
20-feature model and build a 32-feature matrix, failing 113 minutes in:

```
LightGBMError: The number of features in data (32) is not the same as
it was in training data (20).
```

A model and the code that feeds it are one deployable unit. `COMMIT` exists to
enforce that.

## Reproducing without Kaggle

Nothing here is required. The pipeline runs standalone:

```bash
cd ../src
python train_model.py --sample-entities 30000
python pipeline.py predict --country India --shard 0 --shards 5
python pipeline.py merge --output-dir ../../output
```

See `../README.md` for the full sequence and environment requirements.
