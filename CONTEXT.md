# CONTEXT — single source of truth

Authoritative record of this project. **Every number here was measured, not
estimated.** Inference is labelled as inference. Companions: `ps.md` (problem
statement facts), `brainstorm.md` (strategy reasoning).

Last updated: 2026-09-25 15:20.

---

## 1. MISSION

**ML Challenge 2026 — Business Entity Resolution.** Team **Lassi** (solo, Bhuvi).

Given business records from 3 independent sources with noisy, inconsistent
fields and **no shared identifiers**, determine which records refer to the same
real-world business.

- Source 1 is the deduplicated reference. For each S1 entity, find all matching
  records in S2 and S3. An entity may match **zero, one, or many**.
- **Metric: macro-averaged F₀.₅** — computed per S1 entity, then averaged.
  `F_0.5 = (1.25 × P × R) / (0.25 × P + R)`. Precision weighted 2× over recall.
- **Singletons score 1.0 if correctly predicted empty, 0.0 if anything is
  predicted.** Included in the average.
- Deliverables: `matching_results.tsv` (scored) + `candidate_pairs.tsv`
  (audited) + runnable code + methodology doc, as `Lassi_submission.zip`.
- **Winning = a submission that scores.** Not an impressive system.
- **TWO ranked axes, not one** (PS update banner, captured 2026-09-25):
  1. macro F₀.₅ on the private split — maximise;
  2. **mean candidates per Source 1 entity — minimise.** "The approach that
     generates a smaller candidate set per Source 1 entity will be ranked higher
     in the final evaluation beyond the public/private leaderboard."
  Axis 2 is a tiebreak applied after the leaderboard, so score still dominates —
  but the top 500 are all above 0.95, i.e. tightly clustered, which is exactly
  the regime where a tiebreak decides placings. v1 measured **76.8 candidates
  per S1 entity**; v2's figure is unmeasured. See `ps.md` for the full rule and
  the prune-stage design if the number comes back high.

### Hard constraints
| Constraint | Detail |
|---|---|
| External data | **BANNED** — APIs, geocoding, registries, internet augmentation. Disqualification |
| Models | MIT/Apache 2.0 only, ≤8B params. **We use NO pretrained weights** — LightGBM from scratch, so satisfied by construction |
| Reproducibility | **Graded.** Package must regenerate both outputs standalone. `requirements.txt` rebuilt 2026-09-25 from actual imports — the previous file was a whole-machine `pip freeze` that omitted lightgbm, sparse_dot_topn, rapidfuzz and Unidecode |
| Blocking parsimony | **Ranked.** `candidate_pairs.tsv` and the code producing it are reviewed; smaller candidate sets rank higher |
| Deadline | <48h from 2026-09-25. Submissions **unlimited** |

---

## 2. CURRENT STATUS (the dashboard)

| Component | State |
|---|---|
| Pipeline code (9 modules) | ✅ complete, committed, pushed |
| Unit tests | ✅ 28/28 pass |
| Synthetic end-to-end test | ✅ PASS incl. unseen-country path |
| **Training run** | ✅ **macro F₀.₅ = 0.9099**, threshold 0.575 |
| **France predictions** | ✅ 259,452 rows, verified, downloaded |
| **US predictions** | ✅ 663,106 rows, verified, downloaded |
| **India predictions** | ✅ 809,986 rows, verified |
| **Merged output** | ✅ 1,732,544 rows — matching 83 MB, candidates 1.6 GB |
| Official validator | 🟡 running |
| **SUBMITTED** | ✅ **LEADERBOARD 0.890575** |
| Methodology doc | ✅ written (needs final numbers) |
| Submission zip | ❌ not yet assembled |

**Repo:** `github.com/Bhuvilol/AmazonML`, HEAD `867b2f8`, 36 files.
**Kaggle:** user `zeroxBhuvii`. Token expires in ~1.1h → needs
`kaggle auth login --force` when it does (kernels keep running regardless).

---

## 3. THE DATA (all measured)

| Split | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

~2.4 GB TSV. Test pair space **1.7e13** ⇒ blocking is existential.

### Country split
| Country | S1 test | S2 test | S3 test | in train? |
|---|---|---|---|---|
| India | 809,986 | 2,312,565 | 2,405,000 | yes (883,188) |
| US | 663,106 | 1,871,330 | 1,945,701 | yes (1,323,633) |
| **France** | **259,452** | 703,378 | 731,615 | **NO — 15.0% of test** |

### Ground truth
- GT rows **2,206,822 == S1 rows** ⇒ exactly one row per entity
- **Singletons 123,247 = 5.58%** ⇒ all-empty submission scores **0.0558**
- Non-singletons 94.42%, **mean 3.67 matches**, max 11
- Total true pairs **7,638,365**

### Three structural findings
| # | Finding | Evidence |
|---|---|---|
| Q2 | **Perfect mutual exclusivity** — each S2/S3 record belongs to ≤1 S1 entity | 7,638,365 ids, **zero** claimed twice |
| Q3 | **Zero cross-country matches** ⇒ country is a lossless partition | 693,069 pairs checked, 100% same |
| — | **Cross-script problem** | S1 is 0% non-ASCII; S2 15.1%, S3 11.5% |

### Cross-script detail (all 7.6M true pairs)
| S2/S3 name | address | share |
|---|---|---|
| latin / latin | | 79.30% |
| latin / NON-LATIN | | 6.81% |
| **NON-LATIN / latin** | | **11.63%** |
| NON-LATIN / NON-LATIN | | 2.26% |

**Per country** — this is why India is harder:
| | India | US |
|---|---|---|
| name non-Latin | 23.5% | 7.5% |
| **addr non-Latin** | **22.6%** | **0.0%** |
| no Latin bridge | 5.6% | 0.0% |

---

## 4. ARCHITECTURE

```
load → normalise → block (union) → featurise → score → decide → write
```
One country partition at a time (lossless per Q3; bounds memory).
**Country is discovered from data, never hard-coded** — France requirement.

| Module | Role |
|---|---|
| `io_tsv.py` | TSV I/O; tab sep, no quoting, ids as str, empty≠null, polars-version-agnostic |
| `normalize.py` | Vectorised: accent folding, legal-suffix strip, abbrev expansion, **sorted-token** blocking keys |
| `blocking.py` | char_wb 3-gram TF-IDF + `sp_matmul_topn`, union of name & address |
| `features.py` | 20 pairwise features via rapidfuzz |
| `model.py` | LightGBM, **entity-grouped** splits |
| `decide.py` | threshold + singleton gate + exclusivity resolution |
| `pipeline.py` | orchestration, `--country` partials, `merge` |
| `submit.py` | writers enforcing every validator rule; streaming |
| `metric.py` | macro F₀.₅ with singleton convention |

### Tuned parameters (all by measurement)
| Param | Value | Evidence |
|---|---|---|
| `max_df` | 0.01 | 21× faster than none for 1.7pt recall; 0.003 collapses to 0.84 |
| `ngram_range` | (3,3) | faster AND better recall than (2,3) at this max_df |
| `top_n` | 40/40 | 0.9279@20 → 0.9397@30 → 0.9462@40; blocking time FLAT across k |
| `min_sim` | 0.25/0.30 | **inert** — 0.25 vs 0.10 gave identical recall |
| `n_threads` | −1 | sparse_dot_topn treats 0/None as SERIAL (2.7× loss) |
| threshold | 0.575 | swept against macro F₀.₅; curve flat 0.475–0.65 |

### Key design reasoning
- **Union of name AND address blocking** recovers the 11.63% of pairs with a
  non-Latin name but Latin address. Name-only concedes ~14% recall.
- **Country as boolean `same_country`, never one-hot** — generalises to France.
- **Blocking is recall-greedy; decision is precision-biased.** Opposite biases,
  deliberately. For a 4-match entity: all 4 = 1.000, 3 of 4 = 0.9375, all 4 + 1
  false positive = 0.8333. **Missing beats adding.**

---

## 5. RESULTS SO FAR

### v2 Training — COMPLETE 2026-09-25 (`top_n=40`, `max_df=0.50`, + exact-key)
```
macro F0.5 = 0.9218   threshold 0.650   all-empty baseline 0.0579
```
Blocking recall ceiling: **US 0.9757, India 0.9254** (v1: 0.9462 / 0.8796).

| country | positives | candidates | positive rate |
|---|---|---|---|
| India | 96,102 | 2,601,321 | 3.69% |
| US | 101,279 | 2,735,997 | 3.70% |

Exact-key blocking contributed 720,922 pairs (India S2) and 598,548 (US S2);
678 India / 4,177 US sources hit an oversized key (>100) and were **truncated,
not skipped** — the H5 fix.

**Read on this:** +4.6 points of India recall ceiling and +3.0 US, for +1.2
points of local macro F0.5 (0.9099 -> 0.9218). The recall gain converts to score
at roughly 1:4, because recall was a ceiling, not the binding constraint at
every entity. Extrapolating v1's local-to-leaderboard gap (-0.019) puts v2 near
**0.903** — real progress, still far from 0.9859. **The remaining gap is not a
blocking problem.** See section 8.

### v1 Training (Kaggle, 71 min, 100k entities/country)
```
macro F₀.₅ = 0.9099   threshold 0.575   all-empty baseline 0.0582
```
Blocking recall: **US 0.9462, India 0.8796**.
Feature gain: `addr_token_set` 50.6, `name_jaro` 18.9, `nums_token_set` 6.2.

### Test predictions
| stage | entities | candidates | runtime | empty% | mean |
|---|---|---|---|---|---|
| France | 259,452 | 19,482,447 (75.1/ent) | 13 min | **16.11%** | 2.90 |
| US | 663,106 | 50,739,173 (76.5/ent) | 123 min | **6.54%** | 3.33 |
| India | 809,986 | 62,869,162 (77.6/ent) | 223 min | **8.58%** | 3.15 |
| **merged** | **1,732,544** | **133,090,782** | — | — | — |
| *(train)* | | | | *5.58%* | *3.67* |

Country empty-rate ordering is consistent with difficulty: US 6.54% (in train,
Latin) < India 8.58% (in train, 23% non-Latin) < **France 16.11%** (not in
train at all). Train actual singleton rate is 5.58%.

All downloaded files format-verified: correct row counts, 0 malformed rows,
0 self-matches, no spaces after commas.

### ⚠️ OPEN FINDING: France abstains 2.5× more than US
US (6.54%) tracks train (5.58%). France is 16.11%. **US is the control**, so
this is signal not noise: the model is less confident where it has no training
data. Under F₀.₅ abstaining on a matched entity scores 0.0 — same as guessing
wrong — so abstention only pays on true singletons (~5.6%). Estimated cost
**~1 point**. France reruns in 13 min ⇒ testable via a second submission.

---

## 6. MY WRONG PREDICTIONS (6 so far — do not re-introduce)

| # | I predicted | Measurement said |
|---|---|---|
| 1 | Singletons a major lever (~35%) | **5.58%** — a guard, not the lever |
| 2 | Mutual exclusivity buys precision | **0.0000 gain** — 1 conflict in 15,000 |
| 3 | Recall 0.9709 | **0.9278** — toy 369k pool over-stated by 4pts |
| 4 | Numeric blocking closes India gap | **+0.004** — redundant with address blocking |
| 5 | "Address is the bridge" | True for US, **false for India** (22.6% non-Latin addrs) |
| 6 | Timings (×4) | France 30→13min, US 2.5h→2.05h, India 4h→>3h |

**Pattern: plausible mechanism, measurement disagrees.** Measure first, always.

### Corrections to earlier claims
- Chunk 2: predicted two id-list columns → actually **one** (`matched_entity_ids`)
- Chunk 4: inferred no leaderboard → there **is** one (public + private split)
- Claimed precision weighted 4:1 → PS says **2×** (conventional F_β reading)
- Claimed ids zero-padded 5-digit → actually variable length (`S1-925783039`)

### Bugs I introduced and fixed
1. **Index-offset mismatch** — offset predictions but not truth ⇒ bogus 0.4690.
   Pinned by `test_offset_predictions_against_unoffset_truth_scores_zero`.
2. **scipy prunes explicit zeros** in sparse addition ⇒ candidate arrays
   misaligned. Fixed with sorted int64-key gather + assertion.
3. **Ran blocking single-threaded for hours** (`n_threads=0` is serial).
4. `.gitignore` `artifacts/**` didn't match at depth ⇒ 4 MB model nearly committed.
5. Poller treated **auth failure as job completion** ⇒ false "all finished".
6. Renamed a polars kwarg for a local warning ⇒ **broke Kaggle** (older version).

---

## 7. INFRASTRUCTURE — what failed and why

**Local M1 (8 cores, 8 GB) RESET under load.** Memory, not heat: two countries'
TF-IDF matrices + both feature matrices + a `np.vstack` duplicate ≈ 6–7 GB.

**AWS BLOCKED — account on the Free Plan.** Largest permitted instance
`m7i-flex.large` (2 vCPU / 8 GiB) — *smaller than the laptop*. Requires the user
to upgrade to a Paid Plan. Left in place for if that happens: S3 bucket
`lassi-er-920876082653`, IAM role `LassiERInstanceRole`, instance profile.
Best value if upgraded: **`c6a.16xlarge` 64 vCPU/128 GiB @ $1.496/h** (half the
price of the Intel equivalent).

**CHOSEN: Kaggle** — 4 vCPU, **31 GB RAM**, 12h sessions, free.

### Kaggle gotchas (cost ~5 failed runs)
- `enable_internet: true` **ignored** unless account is phone-verified
- **`kernel_sources` silently does not mount** → publish artifacts as a DATASET
- OAuth expires ~3h, CLI does **not** auto-refresh → `kaggle auth login --force`
- Kaggle polars **1.35.2** (local 1.44) → version-agnostic kwargs required
- Large outputs **download partially with no error** → verify by size, retry
- macOS has no `timeout` command (it's `gtimeout`)

### Kaggle assets
| Kind | Name |
|---|---|
| Dataset (data) | `zeroxbhuvii/amazonml-er-data` — 7 files, verified byte-exact |
| Dataset (model) | `zeroxbhuvii/lassi-er-artifacts` — model.txt + threshold.json |
| Kernels | `lassi-er-{train,india,us,france}` |

---

## 7b. LEVER SWEEP — 2026-09-26, all measured, none large

Ran while the v2 shards were executing. Every candidate lever for closing
0.903 -> 0.9859 was measured rather than argued. **None of them is large.**

| lever | measured gain | cost | verdict |
|---|---|---|---|
| corruption inversion (state/ordinal/pad/null) | **+0.0024** recall@40 | 6.5 h re-run | not worth it |
| decision rule (best feasible) | **+0.0042** | free, post-processing | **take it** |
| French normalisation (`R.`->`Rue`, region<->dept) | **+0.0070** top-1 addr cosine | 6.5 h re-run | not worth it alone |
| global threshold | **0** | free | exhausted |
| better model | **~0** | — | AP already 0.9821 |

### Corruption is mechanical but inverting it barely helps
14,754 true pairs sampled. The noise IS rule-based and enumerable: state
spelled-out<->abbreviated **27.34%**, NULL/None literal 7.65%, domain form 5.47%,
zero-padding 5.35%, homoglyph digit (`5anderson`, `Fa1cone`) 2.89%, scrambled
ordinal (`80th`->`80nd`) 1.15%. Inverting all of them raises mean address cosine
**+0.061** — but only **12 net pairs of 8,887** cross `min_sim_addr=0.30`,
because 94.88% were already above it, and recall@40 moves just **+0.0024**.
**The corruptions reduce margin, not reachability.**

### The decision layer: real headroom, but hard to capture
Model average_precision is **0.9821** and the threshold curve is a flat plateau
(0.55–0.725 all within 0.002), so the pairwise ranking is near-saturated and no
global cut does better. Oracle analysis on 3,000 US entities:

| rule | macro F0.5 | captures |
|---|---|---|
| best global threshold | 0.9762 | — |
| oracle best-prefix | 0.9982 | headroom +0.0220 |
| oracle best-subset | 0.9991 | +0.0009 beyond prefix |
| A — expected-F0.5 under independence | 0.9638 | **-57% (LOSES)** |
| B — threshold + singleton gate | 0.9796 | 15% |
| C — relative cut `p_k >= r*p_1` | **0.9805** | **19%** |

**Prefix-oracle ~ subset-oracle** (gap 0.0009): the within-entity ranking is
right, only the cut point is wrong. But feasible rules capture just 15–19% of
that. **Rule A — the per-entity expected-F optimiser I was most confident in —
actively loses**, because LightGBM probabilities are not calibrated and
independence is a poor model of the score distribution. Brainstorm step 3c
failed its exit criterion and is deleted, exactly as specified.

### France: the blocking hypothesis is dead
France abstains at 16.11% vs 6.54% US, and the estimate "France ~ 0.78" made it
look like the biggest single lever (~0.02 overall). Measured directly on 4,000
France test entities vs 120k targets:

```
top-1 address cosine: mean 0.8948  median 0.9088  >=0.5 99.80%  >=0.7 97.08%  zero 0.00%
```

**France blocking is excellent.** 99.8% of entities already have a strong top-1
candidate; none are at zero. French-aware normalisation adds only +0.0070.
The candidates exist — the model scores them low. Cause is under test.

⚠️ **"France ~ 0.78" is an inference, not a measurement.** It comes from
`0.85 x 0.91 + 0.15 x France = 0.8906`, which assumes US+India score 0.91 on
test because that was the local validation figure. If US+India actually score
0.88 on test, France is ~0.95 and there is no France problem. **Do not treat
0.78 as data.**

---

## 8. WHAT REMAINS

1. **India finishes** → download with size verification
2. **Merge locally**: `python -m pipeline merge --output-dir <dir>`
   (verifies every test entity appears exactly once)
3. **Validate**: `python3 utils/validate_submission.py --matching … --candidate …
   --test-dir dataset/test`
4. **Upload `matching_results.tsv`** → first real leaderboard score
5. **Then** the France threshold A/B (13 min rerun, ~1 point expected)
6. Assemble `Lassi_submission.zip`: `output/` + `code/business_entity_resolution/`
   + filled `Documentation_template.md`
7. Update methodology doc with final numbers

### Untested lever if time allows
India recall 0.880 vs US 0.946. Only 5.6% of India pairs have no cross-script
bridge, but 11.5% are missed ⇒ **~6% are ranking failures** (true match exists
in-script but falls outside top-40), not signal failures. Higher `top_n` is the
untested lever there — NOT another signal (numeric was measured and rejected).

### ACTUAL LEADERBOARD SCORE: **0.890575**

| reference | score |
|---|---|
| all-empty baseline | 0.0582 |
| local validation (US+India) | 0.9099 |
| **leaderboard (US+India+France)** | **0.890575** |

Validation was 2 points optimistic — well calibrated for this metric.

### The gap decomposes to France (inference, but well supported)
Validation could only cover US and India. If those transfer cleanly to test:
```
0.85 × 0.91 + 0.15 × France = 0.8906   =>   France ≈ 0.78
```
**France scores ~0.78 vs ~0.91 for trained countries — a 13-point gap on 15%
of the test set.** Consistent with the abstention evidence (France 16.11% vs
US 6.54%, true singleton rate 5.58%).

Assumption: US/India test performance ≈ their validation performance. Not
verifiable directly, but the abstention data independently points the same way.

**Headroom: lifting France 0.78 → 0.85 would add ~0.010 to the total.**
