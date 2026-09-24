# CONTEXT — quick reference

Dense factual reference so nothing measured gets lost or re-derived.
**Every number here was measured, not estimated.** Inference is labelled.
Companions: `ps.md` (problem facts), `brainstorm.md` (strategy).

---

## 1. Scale (measured)

| Split | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

~2.4 GB of TSV. Test pair space **1.7e13** → blocking is existential.

### Country split (test S1: France = 15.0%)
| Country | S1 test | S2 test | S3 test |
|---|---|---|---|
| India | 809,986 | 2,312,565 | 2,405,000 |
| US | 663,106 | 1,871,330 | 1,945,701 |
| France | 259,452 | 703,378 | 731,615 |

train S1: US 1,323,633 / India 883,188. **No France in train.**

---

## 2. Ground truth (measured, full file)

- GT rows **2,206,822 == S1 rows** → exactly one row per entity, no ambiguity
- **Singletons 123,247 = 5.58%** → all-empty submission scores **0.0558**
- Non-singletons 2,083,574 (94.42%), **mean 3.67 matches**, max 11
- Total true pairs **7,638,365**
- Cardinality: 1→5.4%, 2→17.0%, 3→24.1%, 4→21.9%, 5→14.6%, 6→7.5%, 7→2.9%, 8+→1.1%

---

## 3. Structural findings (the levers)

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| Q2 | **Perfect mutual exclusivity** | 7,638,365 distinct ids, **0** claimed by >1 S1, max=1 | Each S2/S3 record belongs to ≤1 S1. Resolve contested records to best scorer → free precision. NOT YET IMPLEMENTED |
| Q3 | **Zero cross-country matches** | 693,069 pairs checked, 100% same-country | Country = lossless hard partition. Solves memory |
| Q4 | One GT row per entity | row counts equal | No missing-vs-singleton ambiguity |

### Cross-script (measured on all 7,638,365 true pairs)
| S2/S3 name | address | pairs | share |
|---|---|---|---|
| latin | latin | 6,057,423 | 79.30% |
| latin | NON-LATIN | 520,161 | 6.81% |
| **NON-LATIN** | latin | 888,096 | **11.63%** |
| **NON-LATIN** | **NON-LATIN** | 172,685 | **2.26%** |

S1 names **0%** non-ASCII. S2 names **15.1%**, S3 **11.5%**, S2 addr **9.4%**.
→ **Address blocking mandatory** (bridges 11.63%). Transliteration rejected: only 2.26% hole.

---

## 4. Locked decisions (measured, not guessed)

| Decision | Value | Evidence |
|---|---|---|
| Partition | by country | Q3: zero cross-country |
| Vectoriser | char_wb **3-grams**, `max_df=0.01` | 16s vs 338s baseline, recall 0.9709 vs 0.9883 |
| `top_n` | **20** name + 20 addr | 20→30 buys +0.47% recall for +52% pairs |
| Candidate union | name ∪ addr, **scores kept separate** | name_cos=0 is informative (cross-script regime) |
| Transliteration | **NO** | 2.26% hole, diluted further by macro averaging |
| Pretrained models | **NONE** | LightGBM from scratch → MIT/Apache constraint trivially met |

### Blocking sweep (20k S1 × 369k targets)
| config | secs | recall |
|---|---|---|
| no max_df (2,3) | 338 | 0.9883 |
| max_df=0.10 (2,3) | 149 | 0.9874 |
| max_df=0.01 (2,3) | 25 | 0.9667 |
| **max_df=0.01 (3,3)** | **16** | **0.9709** |
| max_df=0.003 (3,3) | 9 | 0.8368 (cliff) |

### top_n sweep (blocking time is FLAT ~16s across all)
| top_n | recall | cand/entity | pairs @1.73M |
|---|---|---|---|
| 5 | 0.9119 | 8.4 | 14M |
| 10 | 0.9561 | 17.9 | 31M |
| **20** | **0.9709*** | **37.4** | **65M** |
| 30 | 0.9756* | 57.0 | 99M |

\* measured against a 369k **toy** target pool. **Realistic ceiling against the
full 6,186,873-record US train pool is 0.9278** (64,215/69,211). Always quote
the realistic number; the toy pool over-states recall by ~4.3 points because
10x more distractors compete for the same top-20 slots.

---

## 4b. Performance facts (hard-won)

| Fact | Detail |
|---|---|
| **`sparse_dot_topn` n_threads trap** | It does `n_threads = n_threads or 1`, so **0 or None runs SERIAL**. Must pass **`-1`**. Cost of the bug: 2.7x on this machine |
| Threading gain (M1, 8 cores) | 243s → 89s = **2.7x**, byte-identical output (394,120 nnz both). Not 8x — memory-bandwidth bound |
| Vectorising is NOT the bottleneck | `TfidfVectorizer.fit_transform` on **6,186,873 docs = 41s**. The matmul dominates |
| Blocking cost (threaded) | 20k source x 6.19M targets = **89s per field** ⇒ ~4.45 ms/source-row at 6.19M targets; scales ~linearly in both source rows and target count |
| **macOS LightGBM** | needs `brew install libomp` or `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib`. Must go in the submission README |

### Full-run projection (threaded, top_n=20, max_df=0.01, 3-grams)
| Partition | S1 | targets | est. both fields |
|---|---|---|---|
| US test | 663,106 | 3,817,031 | ~1.0 h |
| India test | 809,986 | 4,717,565 | ~1.5 h |
| France test | 259,452 | 1,434,993 | ~9 min |
| **test total** | | | **~2.6 h** |

Train blocking on a ~150k entity sample adds ~25 min. **Fits in 48h — no EC2 needed.**
Escalation trigger did NOT fire: the bottleneck was a config bug, not hardware.

---

## 5. Features — 20, throughput 0.19M pairs/sec (65M ≈ 6 min)

Separation (pos_mean − neg_mean), US sample:
`nums_exact` **.615** | `nums_token_set` .593 | `addr_cos` .511 | `name_token_set` .429 |
`name_token_sort` .417 | `addr_token_set` .420 | `name_ratio` .405 | `name_tok_diff` **−.592**

**Exact address-number match is the single strongest feature**: 63.1% of positives vs 1.6% of negatives.

⚠️ `tgt_is_s3` showed .439 separation = **SAMPLING ARTIFACT**. Distractors were
`head()` of concat(S2,S3) ⇒ nearly all S2, positives ~50/50. Distractor pools
must be random or full-population. Do not trust that number.

---

## 6. Validator rules (read from `utils/validate_submission.py`)

- Header **exact**: `source1_entity_id\tmatched_entity_ids` / `...candidate_entity_ids`
- **Every row needs a tab**, even empty ones → `S1-1\t` not `S1-1`
- **No space after commas** — ids are NOT stripped; `S2-1, S2-2` ⇒ ` S2-2` ⇒ prefix error
- One row per test S1 entity; no dup rows; no dup ids in a list; no `S1-` self-match
- ID-existence check **OFF by default**, and is a *diagnostic* not a gate
- matching ⊆ candidates is a **WARNING, never a failure**
- Must be valid UTF-8

---

## 7. Environment

- Python **3.11.16** venv (system 3.14 too new). `sagemaker<3`, `sqlalchemy<2.1` mandatory
- Deps: polars, scikit-learn, scipy, lightgbm, rapidfuzz, sparse-dot-topn (all BSD/MIT/Apache)
- Machine: **M1, 8 cores, 8 GB RAM**
- AWS ap-south-1, IAM `bhuvi-dev`. **GPU quota 0, requests `CASE_OPENED`** (human review, not arriving)
- EC2 standard **16 vCPU available**; SageMaker notebook max `ml.r5.xlarge` = **4 vCPU = half the M1**
- Escalation trigger: OOM/slow ⇒ EC2 `r5.4xlarge`. Poor score ⇒ NOT a hardware problem
- Repo: `github.com/Bhuvilol/AmazonML`, git identity `Bhuvilol <bhabeshcse@gmail.com>`, no Claude attribution, commits 5-15 words

---

## 8. Corrections to my own earlier claims (do not re-introduce)

1. Chunk 2: predicted **two** id-list columns → actually **one** (`matched_entity_ids`, S2+S3 mixed)
2. Chunk 4: inferred **no leaderboard** → there **is** one (public subset + private remainder)
3. Claimed precision weighted **4:1** → PS says **2×** (conventional F_β reading)
4. Claimed ids **zero-padded 5-digit** → actually variable length (`S1-925783039`)
5. **Over-weighted singletons** (illustrated ~35%, actually **5.58%**) — abstain gate is a guard, not the main lever
6. **Under-weighted cross-script**; it is a bigger problem than France
7. Blocking alignment bug: **scipy prunes explicit zeros in sparse addition** — `zeros + X` collapses to X's own pattern. Fixed with sorted int64-key gather
8. Ran blocking **single-threaded for hours** without noticing — `n_threads=0` is silently serial in sparse_dot_topn
9. Recall 0.9709 was measured on a **toy 369k target pool**; realistic is **0.9278**. Never quote toy-pool numbers

---

## 9. Build state

| File | State |
|---|---|
| `src/metric.py` | ✅ 27/27 tests, PS example = 0.714 |
| `src/io_tsv.py` | ✅ works on real 1.7M-row files |
| `src/submit.py` | ✅ **official validator PASS** on 1,732,544 rows |
| `src/normalize.py` | ✅ verified on real noisy rows |
| `src/blocking.py` | ✅ tuned; alignment bug fixed + asserted |
| `src/features.py` | ✅ 20 features, 0.19M pairs/sec |
| `src/model.py` | ✅ LightGBM, entity-grouped split, calibration report |
| `src/decide.py` | ✅ threshold + singleton gate + **exclusivity resolution** (unit-verified) |
| `src/run_pipeline.py` | ❌ next |

All-empty baseline written and validator-PASS → submittable now, scores ≈0.056,
and reveals the public-subset singleton rate for free (submissions are unlimited).

---

## 9b. FIRST REAL RESULT (US, 20k entities, full 6.19M target pool)

| Metric | Value |
|---|---|
| **macro F_0.5 (validation)** | **0.9313** |
| all-empty baseline | 0.0508 |
| pair recall ceiling | 0.9278 |
| best threshold | **0.600** (above 0.5, as F_0.5 predicts) |
| pairwise AP | 0.992 |
| blocking (threaded) | 325s for 20k x 6.19M |

Threshold curve is FLAT near the peak (.55→.930, .65→.931, .75→.930) ⇒ robust, low overfit risk.

**Score > recall ceiling is NOT a bug**: ceiling is pair-level, score is per-entity macro
(3-of-4 matches still scores 0.9375; correct singletons score 1.0).

### Feature gain % (differs from raw separation!)
`nums_token_set` **33.6** | `name_ratio` 20.3 | `addr_cos` 10.0 | `addr_token_set` 9.7 |
`addr_token_sort` 7.8 | `name_token_sort` 5.5 | `name_jaro` 4.3 | `name_token_set` 2.8 |
`nums_exact` **0.9** (highest raw separation 0.615, but subsumed by nums_token_set)

`tgt_is_s3` absent from top-10 ⇒ the earlier 0.439 separation really was the sampling artifact.

### ⚠️ Prediction I got WRONG
**Mutual exclusivity gave ZERO gain** (0.9313 both ways; only 1 contested claim in ~15,000).
At threshold 0.6 the model is precise enough that conflicts are vanishingly rare. Kept as a
cheap correctness guarantee, but it is NOT a differentiator. Do not re-inflate this claim.

**Binding constraint is now RECALL (0.9278), not the model (AP 0.992).**

---

## 10. Open / next

1. ~~realistic recall ceiling~~ → **0.9278** measured. Blocking 20k x 6.19M took **713s** — naive full-test projection ~10h, needs fixed-vs-variable cost split before deciding optimise-vs-escalate.
2. `model.py` — LightGBM, group split on S1 entity, stratify singletons
3. `decide.py` — 3a global threshold → 3b singleton gate → 3c exploit mutual exclusivity
4. `run_pipeline.py` end-to-end, submit
5. Reserved final hours: zip package + `Documentation_template.md`
6. Deadline **<48h**, submissions **unlimited**
