# Business Entity Resolution — Team Lassi

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from
the provided training and test data. No pretrained model weights, no external
data, no network access at any stage.

---

## 1. Environment

Python **3.11** is required. Newer CPython releases (3.13+) are ahead of parts
of the ML stack; 3.11 is what this was built and tested against.

```bash
# macOS only: LightGBM links against the OpenMP runtime and will fail at
# import with "Library not loaded: @rpath/libomp.dylib" without it.
brew install libomp

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import lightgbm, polars, rapidfuzz, sparse_dot_topn; print('ok')"
```

### Data layout

The pipeline reads the organisers' layout unchanged:

```
student_resource/
└── dataset/
    ├── train/  train_source{1,2,3}.tsv, train_ground_truth.tsv
    └── test/   test_source{1,2,3}.tsv
```

---

## 2. Reproduce end to end

```bash
cd src

# 1. Train the pairwise matcher and select the decision threshold.
#    30,000 entities PER COUNTRY is the figure that produced the submitted
#    model -- not a placeholder. Blocking each sampled entity against the full
#    6.19M-record target pool is the expensive part, so this is where the
#    runtime goes (~2.5 h on 4 vCPU), not in fitting the trees.
#    Writes artifacts/model.txt and artifacts/threshold.json
python train_model.py --sample-entities 30000

# 2. Generate both submission files for the test set.
#    The threshold is read from artifacts/threshold.json automatically,
#    so the two stages cannot drift apart.
python pipeline.py predict --model ../artifacts/model.txt --output-dir ../../output
```

### Running it in pieces

The single command above needs one machine to hold the whole run. Test
prediction is ~4.5 h of CPU, so we ran it split by country and shard and merged
the parts. Any subset works; `merge` reassembles whatever partials it finds:

```bash
# One country at a time.
python pipeline.py predict --country France --output-dir ../../output

# Or shard a large country across machines (5 ways here).
python pipeline.py predict --country India --shard 0 --shards 5 --output-dir ../../output

# Reassemble. Writes matching_results.tsv and candidate_pairs.tsv, one row per
# test Source-1 entity, and fails loudly if any entity is missing.
python pipeline.py merge --output-dir ../../output
```

**Merge collects by pattern** (`matching_results_*.tsv`), so run it against a
directory holding one run's partials only. Pointing it at a directory that also
contains an earlier run's files silently mixes both generations into an output
that still has the right row count and still passes the official validator.

Then validate with the organisers' script, from `student_resource/`:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Runtime on an 8-core Apple M1 with 8 GB RAM: roughly **2.5-3 hours** for the
full test set, dominated by candidate generation.

---

## 3. Pipeline

```
load ─► normalise ─► block (candidates) ─► featurise ─► score ─► decide ─► write
```

Processing is **one country partition at a time**. True matches never cross
countries (verified on 693,069 training pairs, zero exceptions), so the
partition is lossless, and it keeps peak memory within 8 GB. Country labels are
**discovered from the data, never hard-coded** — the test set contains France,
which does not appear in training.

| Module | Responsibility |
|---|---|
| `io_tsv.py` | TSV I/O. Tab separator, no quoting, ids as strings, empty ≠ null |
| `normalize.py` | Vectorised text normalisation; accent folding, legal-suffix and address-abbreviation handling, sorted-token blocking keys |
| `blocking.py` | Character n-gram TF-IDF + top-k sparse multiply; union of name and address candidates |
| `features.py` | 20 pairwise similarity features via rapidfuzz |
| `model.py` | LightGBM binary classifier, entity-grouped validation split |
| `decide.py` | Threshold, singleton gate, and mutual-exclusivity resolution |
| `pipeline.py` | Orchestration and CLI |
| `submit.py` | Writes both outputs, enforcing every validator rule at write time |
| `metric.py` | Macro F_0.5 with the singleton convention |

---

## 4. Tests

```bash
python -m pytest ../tests -q
```

`tests/test_metric.py` verifies the scorer against the worked example in the
problem statement (0.714) and pins the precision-asymmetry values that drive
the decision stage.

---

## 5. Key configuration

Defaults live in `pipeline.BlockingConfig` and were chosen by measurement, not
by feel:

| Setting | Value | Why |
|---|---|---|
| `max_df` | 0.01 | 21x faster than no pruning for ~1.7 points of recall. 0.003 collapses recall to 0.84 |
| `ngram_range` | (3, 3) | Faster *and* higher recall than (2, 3) at this `max_df` |
| `top_n_name` / `top_n_addr` | 20 / 20 | 20→30 buys +0.5 points of recall for +52% candidate volume |
| `n_threads` | **-1** | `sparse_dot_topn` treats `0`/`None` as serial; `-1` is worth 2.7x here |

---

## 6. Licences

No pretrained model weights are used. The matcher is LightGBM trained from
scratch on the provided training data only, so the MIT/Apache 2.0 and 8B
parameter constraints are satisfied by construction.

| Package | Licence |
|---|---|
| numpy, scipy, scikit-learn, pandas | BSD-3-Clause |
| polars, lightgbm, rapidfuzz | MIT |
| sparse-dot-topn | Apache-2.0 |
