# Business Entity Resolution — Team Lassi

**ML Challenge 2026** · Final leaderboard score **0.937** (macro-averaged F₀.₅)

Matching business records across three independent sources with noisy,
inconsistent fields and no shared identifiers.

```
0.890575  →  0.893467  →  0.937
   v1          hybrid       final
```

---

## The problem

Given ~1.7M Source-1 business records and ~10M Source-2/3 records across three
countries, find every Source-2/3 record referring to the same real-world
business as each Source-1 entity. An entity may match zero, one, or many.

Scored by **macro-averaged F₀.₅ per entity** — precision weighted 2× over
recall, with singletons scoring 1.0 for a correct empty prediction and 0.0 for
any prediction.

Two things make it hard:

- **Names and addresses are corrupted** — abbreviations, legal-suffix
  inconsistency, typos, homoglyphs, word-order transpositions, transliteration
  into and out of five Indic scripts, dropped address components.
- **France appears only in the test set.** The pipeline must generalise to a
  country it has never seen, so nothing may hard-code or one-hot the country.

---

## Architecture

```
raw TSV
   │
   ├─ normalise ─────── transliterate · fold accents · strip punctuation
   │                    expand abbreviations · drop legal suffixes · sort tokens
   │
   ├─ block ─────────── name char-3gram TF-IDF top-40   ┐
   │                    addr char-3gram TF-IDF top-40   ├─ union
   │                    exact name key (hash join)      │
   │                    exact addr key (hash join)      ┘
   │                    → ~88 candidates / entity, reduction ratio 0.99998
   │
   ├─ featurise ─────── 20 pairwise-absolute
   │                    13 exact-match + IDF
   │                    10 competition (rank / margin / share)
   │
   ├─ score ─────────── LightGBM binary GBDT, 43 features, AP 0.9821
   │
   └─ decide ────────── threshold 0.675 · mutual-exclusivity resolution
                        → matching_results.tsv + candidate_pairs.tsv
```

**Recall-greedy blocking, precision-biased decision.** The two stages are
deliberately biased in opposite directions: a true pair not proposed by blocking
can never be recovered, while precision is purchasable at the threshold. This is
the single most important structural decision in the pipeline.

---

## Results

| | US | India |
|---|---|---|
| Blocking recall ceiling | 0.9757 | 0.9254 |
| Locked-holdout macro F₀.₅ | 0.9620 | 0.9266 |
| Pairwise precision / recall | 0.9875 / 0.8777 | 0.9606 / 0.8088 |

Improvement over the 20-feature baseline, measured on a locked holdout with
thresholds selected on one half and reported on the other:

| Configuration | India | US |
|---|---|---|
| 20 features | 0.8975 | 0.9470 |
| + 13 IDF / exact-match | 0.9198 | 0.9567 |
| + 10 competition | **0.9266** | **0.9620** |

---

## Three findings worth reading the docs for

**1. Address blocking carries the system, not name blocking.**
Measured per-strategy on held-out data, address TF-IDF contributes **13,907
unique true pairs** to India's candidate set against name's **1,497**. 11.63%
of true pairs have a non-Latin name *and* a Latin address, so name similarity is
structurally zero and only the address can propose them. Name-only blocking
concedes ~14% of recall.

**2. The model was missing a whole class of signal.**
All 20 original features describe a pair *in isolation*. None could express
whether a candidate was the best of forty or the thirty-seventh of forty, or
whether the tokens two records share are *rare* or *ubiquitous*. Adding those
two classes was worth **+0.0291 India / +0.0150 US** and cost nothing — every
input was already sitting in the blocking output.

**3. Four good ideas measured out as not worth shipping.**
Learned normalization recovered 9 blocked pairs. Reverse retrieval worked but
cost 25–414 candidates per recovered pair. 236 cardinality-aware decision
policies were all negative. A per-entity expected-F optimiser scored −0.0131.
See [§11](Lassi_submission/Documentation_template.md#11-experimental-log).

---

## Repository layout

```
Lassi_submission/
  Documentation_template.md        full methodology — 17 sections, 1,700 lines
  code/business_entity_resolution/
    src/                           the pipeline (10 modules)
    tests/                         28 unit tests + synthetic end-to-end harness
    kaggle/                        execution harness, 10 pinned notebooks
    README.md                      environment and reproduction steps
    requirements.txt               9 pinned dependencies

student_resource/                  provided data + official validator

CONTEXT.md                         measured facts and running state
LEARNINGS.md                       22 transferable heuristics
ps.md                              problem statement, annotated
```

### `src/` modules

| Module | Responsibility |
|---|---|
| `metric.py` | Exact competition metric. Written and tested **first** — everything else is calibrated against it |
| `io_tsv.py` | TSV I/O with every trap handled once: tab separator, quoting disabled, IDs as strings, empty ≠ null |
| `normalize.py` | Transliteration, folding, abbreviation expansion, blocking keys |
| `blocking.py` | TF-IDF vectorisation, sparse top-k, exact-key hash join, union assembly |
| `features.py` | All 43 features in three groups |
| `model.py` | LightGBM training, entity-grouped splits, calibration diagnostics |
| `decide.py` | Threshold selection, mutual-exclusivity resolution |
| `submit.py` | Streaming output writer enforcing every rejection rule |
| `pipeline.py` | CLI: `train` / `predict` / `merge` |
| `train_model.py` | Training entry point |

---

## Quickstart

```bash
cd Lassi_submission/code/business_entity_resolution
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python train_model.py --sample-entities 30000          # ~2.5 h on 4 vCPU
python pipeline.py predict --output-dir ../../output   # ~5 h, shardable
python pipeline.py merge   --output-dir ../../output
```

Validate before submitting:

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

Full instructions, sharding, and environment notes:
[`code/business_entity_resolution/README.md`](Lassi_submission/code/business_entity_resolution/README.md)

---

## Tests

```bash
pytest tests/ -q          # 28 tests, < 1 s
```

Plus a synthetic end-to-end harness that generates a small dataset containing
every noise pattern the problem statement names — **and a third country present
only in its test split**, so the unseen-country path is exercised on every run:

```bash
python tests/make_synthetic.py --out /tmp/synth
# then: train → 3-country predict → merge → official validator
```

This harness caught two defects that would otherwise have surfaced only after
multi-hour hosted runs.

---

## Constraints observed

- **No external data.** No APIs, geocoding, business registries, or internet
  augmentation. Every mapping and abbreviation list is either hand-written
  domain knowledge in code or mined from the provided training data.
- **No pretrained weights.** LightGBM trained from scratch, so the MIT/Apache
  and ≤8B-parameter rules hold by construction.
- **Country never encoded.** Used only as a lossless partition key — verified by
  measuring that no true match crosses countries.

---

## Reproducibility notes

Every Kaggle notebook pins the repository to an exact commit and asserts the
checkout succeeded. This exists because it once did not: a feature change pushed
while shards were queued caused one to load a 20-feature model and build a
32-feature matrix, failing 113 minutes in.

**A model and the code that feeds it are one deployable unit.**

Further engineering defects — and what each one teaches — are catalogued in
[§13](Lassi_submission/Documentation_template.md#13-engineering-defects-found-and-fixed)
and [`LEARNINGS.md`](LEARNINGS.md).
