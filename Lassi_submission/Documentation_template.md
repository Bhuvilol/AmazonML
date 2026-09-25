# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Lassi
**Team Members:** Bhuvi
**Submission Date:** 2026-09-25

---

## 1. Executive Summary

A three-stage pipeline — character n-gram TF-IDF candidate generation, a
LightGBM pairwise matcher over 20 similarity features, and a decision stage
tuned directly against macro F_0.5 — run independently per country partition.
The two contributions we consider non-obvious are **blocking on the union of
name *and* address similarity**, which recovers the 11.63% of true pairs whose
Source-2/3 name is in a non-Latin script that character n-grams cannot match at
all, and **tuning the decision threshold against macro F_0.5 itself** rather
than a pairwise proxy, which places it at 0.60 rather than 0.50 because the
metric penalises a false merge roughly twice as hard as a miss.

---

## 2. Methodology

### 2.1 Problem Analysis

Every design decision below follows from a measurement on the training data
rather than an assumption. The measurements that changed our approach:

| Finding | Measurement | Consequence |
|---|---|---|
| **Singletons are rare** | 123,247 / 2,206,821 = **5.58%** | An all-empty submission scores only 0.056. Abstention is a guard, not the main lever — our first design over-weighted it |
| **Matches never cross countries** | 693,069 pairs checked, **zero** exceptions | Country is a *lossless* hard partition. Cuts the pair space and bounds memory |
| **Matches are mutually exclusive** | 7,638,365 distinct Source-2/3 ids, **zero** claimed by more than one Source-1 entity | A global constraint we enforce, though it proved to bind rarely in practice (see §5) |
| **Source 1 is 100% ASCII; Source 2/3 are 11–15% non-Latin** | Devanagari and Tamil business names | Character n-grams score exactly 0.0 across that boundary. Name-only blocking would concede ~14% of recall |
| **Only 2.26% of pairs lack a Latin bridge** | both name *and* address non-Latin | Transliteration was rejected: the irreducible hole is small and macro averaging dilutes it further |
| **Cardinality** | mean 3.67 matches, max 11, mode 3–4 | Shapes the precision/recall trade (see §4) |

Country distribution confirms the generalisation challenge: training is US
(1,323,633) and India (883,188); the test set adds **France, 259,452 entities —
15.0% of the test Source-1 population** — with no training signal whatsoever.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** Union blocking over name *and* address to bridge
cross-script records, combined with a decision stage optimised directly against
the macro F_0.5 objective rather than a pairwise surrogate.

```
load → normalise → block (union) → featurise → score → decide → write
```

Processing is one country partition at a time. Country labels are **discovered
from the data, never hard-coded** — the pipeline has no branch naming US, India
or France, and an unseen label is simply another partition. This satisfies the
open-set requirement by construction rather than by special-casing.

**Handling France without French training data.** Rather than guess at French
conventions, we made the pipeline structurally language-agnostic:

* Character n-grams instead of word tokens — no vocabulary to be missing.
* Accent folding (`é→e`, `ç→c`) applied uniformly, so diacritics never split a
  match.
* Country is never encoded as a *value*. Where a country signal is useful, it
  appears as the boolean "same country", which generalises to any unseen label
  with no retraining and no fallback path. One-hot encoding would produce an
  all-zero row for France.
* Numeric address tokens, which are script-invariant, carry substantial weight.

---

## 3. Candidate Generation (Blocking)

The upper bound on recall, and the stage we spent most effort measuring.

- **Blocking keys used:** character n-gram (3,3, `char_wb`) TF-IDF cosine over
  two independently constructed keys:
  1. **name key** — lowercased, accent-folded, punctuation-stripped, legal
     suffixes removed (`Pvt`, `Ltd`, `Inc`, `SARL`, …), tokens **sorted**
  2. **address key** — as above plus abbreviation expansion (`Rd→road`), tokens
     sorted

  Token sorting is deliberate: the problem statement promises word-order
  transpositions and address component reordering, and sorting makes the key
  invariant to both. `Sun Constructions Pvt. Ltd.` and
  `SUN CONSTRUCTIONS PRIVATE LIMITED` produce the identical key
  `constructions sun`.

  Top-k is retained per source row using a sparse top-n multiply, so the full
  product is never materialised.

-  **Exact-key blocking, added alongside the cosine top-k.** The ranked top-k
  has a structural failure: a common name has hundreds of key-mates, so a true
  match with a *byte-identical* normalised key can still fall outside the top 40
  and be lost. We found `Office of Housing` / `Office Of Housing` — Jaccard 1.00
  — being missed for exactly this reason. Exact-key blocking hash-joins on the
  normalised key and emits every colliding pair as a candidate, turning a
  ranking into a **guarantee**. Oversized key groups are **truncated, not
  skipped**: an early version skipped groups above 100, which silently discarded
  precisely the common-name cases the mechanism exists to catch.

  Contribution on the training partitions: 720,922 extra India pairs and
  598,548 US pairs; 678 India / 4,177 US sources hit an oversized key.

- **Recall ceiling, measured against the full target pool:**

  | country | target pool | recall ceiling | candidates/entity | reduction ratio |
  |---|---|---|---|---|
  | US | 6,186,873 | **0.9757** | 91.2 | 0.99998526 |
  | India | 4,133,346 | **0.9254** | 86.7 | 0.99997902 |

  Every figure is against the **full** pool, not a sample. This distinction cost
  us real score: an early measurement against a 369k sampled pool reported 0.971
  and was over-optimistic by ~4 points, because 10× fewer distractors compete
  for the same top-k slots. Any blocking parameter tuned on a sample must be
  re-validated at full scale.

- **`max_df` — a pruning setting we initially got wrong.** `max_df` drops
  n-grams appearing in more than the given fraction of documents. We first set
  `0.01`, justified as "a 21× speedup for 1.7 points of recall" — a figure
  measured on the 369k sample. Re-measured at full scale the trade was **2.6
  points**, not 1.7, and at `0.01` roughly 70% of each name's n-grams were being
  discarded. Relaxing to `max_df=0.50` recovered those points:

  | config | US ceiling | India ceiling | time |
  |---|---|---|---|
  | `top_n=40`, `max_df=0.01` | 0.9464 | 0.8863 | 316 s |
  | + exact-key | 0.9531 | 0.9093 | 376 s |
  | `top_n=100` + exact + `max_df=0.50` | 0.9825 | 0.9486 | 1412 s |
  | **`top_n=40` + exact + `max_df=0.50`** (shipped) | **0.9757** | **0.9254** | ~600 s |

  **`top_n` was deliberately kept at 40.** Raising it to 100 buys ~0.7 further
  points of ceiling but roughly doubles both compute and the candidate set. The
  two knobs act on different axes — `top_n` controls candidate *volume*,
  `max_df` controls candidate *quality* — and since the challenge ranks smaller
  candidate sets higher, spending on quality rather than volume is the correct
  trade.

- **How we ensured true matches were not lost — the union, not the
  intersection.** This is the single most important choice in the pipeline.
  Measured over all 7,638,365 training pairs:

  | Source-2/3 name | address | share of true pairs |
  |---|---|---|
  | Latin | Latin | 79.30% |
  | Latin | non-Latin | 6.81% |
  | **non-Latin** | Latin | **11.63%** |
  | non-Latin | non-Latin | 2.26% |

  For that 11.63%, name similarity is structurally zero — the strings share no
  characters — and only the address proposes the pair. Name-only blocking
  discards them permanently.


---

## 4. Matching Model

**Features used** (32 total: 20 pairwise-absolute + 12 competition):

- **Name features:** token-set ratio, token-sort ratio, Levenshtein ratio,
  Jaro-Winkler, TF-IDF character n-gram cosine
- **Address features:** the same five, computed on the normalised address
- **Numeric:** token-set ratio over address digit runs, exact match of the
  sorted digit string, both-present indicator. Digits survive transliteration
  unchanged, which makes them the most reliable cross-script signal
- **Structural:** name/address length ratios, token-count differences
- **Regime flags:** target source (S2/S3), and whether the target's name or
  address is non-Latin. This last group matters: for a cross-script pair every
  name feature is ~0, and the flag lets the model learn *"ignore name
  similarity here, trust the address"* as a regime rather than reading the zero
  as disagreement

Order-insensitive scorers are emphasised throughout, matching the transposition
noise the problem statement describes.

### Competition features — the single largest modelling gain we found

The 20 features above share a blind spot: every one describes a pair **in
isolation**. None of them can express whether a candidate is *the best of forty*
or *the thirty-seventh of forty*. A 0.7 name cosine means something completely
different as the strongest option in a weak field than as an also-ran behind
several near-identical rivals, and no absolute feature can tell those apart.

We added 12 features describing each pair **relative to the others competing for
the same Source-1 entity**, all derived from the candidate arrays blocking has
already produced, at no extra computational cost:

| group | features |
|---|---|
| rank within the entity's list | `rank_name`, `rank_addr`, `rank_comb` |
| gap to the entity's best | `marg_name`, `marg_addr`, `marg_comb` |
| ratio to the entity's best | `rel_name`, `rel_addr` |
| field shape | `n_cands`, `share_comb` |
| reverse direction | `tgt_degree`, `tgt_margin` |

Measured A/B on 6,000 US training entities, 2,400 held out, identical split and
hyperparameters:

| feature set | macro F_0.5 | best threshold |
|---|---|---|
| 20 pairwise-absolute | 0.9800 | 0.775 |
| **+ 12 competition** | **0.9894** | 0.825 |

**+0.0094** — more than double any other lever we tested, including candidate
generation changes costing hours of compute. The importances explain why:

| feature | LightGBM gain |
|---|---|
| **`share_comb`** | **626,234** |
| `rank_comb` | 51,865 |
| `addr_cos` | 44,291 |
| `addr_token_set` | 22,437 |

`share_comb` — a pair's share of its entity's total similarity mass — carries
roughly fourteen times the gain of the strongest string-similarity feature and
dominates the model outright.

A second round of 8 further competition features (second-best level, top-1
dominance, a mutual-best flag, log target degree, mean field strength) scored
**+0.0092** — indistinguishable from the 12-feature set — so they were dropped
rather than carried. Complexity that does not pay rent is removed.

**Implementation note.** These features need an entity's *entire* candidate
list, while scoring is chunked at 2M pairs to bound memory. Computing them per
chunk would cut entities across chunk boundaries and silently corrupt every rank
and share, producing a plausible-looking model that is quietly wrong. They are
therefore computed once over the whole partition, before chunking.

**Model type:** LightGBM (gradient-boosted trees), binary objective. Chosen
because the inputs are ~20 dense bounded tabular scores — precisely where GBDTs
are strongest — and because it trains in minutes on CPU and uses **no
pretrained weights**, so the MIT/Apache 2.0 and 8B-parameter constraints are
satisfied by construction rather than by argument.

Validation splits are **grouped by Source-1 entity**. Splitting by pair would
leak, since two pairs from the same entity share a Source-1 record and often
near-duplicate Source-2/3 records.

**Threshold selection method:** direct macro F_0.5 optimisation on held-out
entities.

This is worth spelling out, because it is where the metric's asymmetry is
actually exploited. For an entity with 4 true matches:

| Prediction | Precision | Recall | F_0.5 |
|---|---|---|---|
| all 4 correct | 1.00 | 1.00 | **1.0000** |
| 3 of 4, no false positive | 1.00 | 0.75 | **0.9375** |
| all 4 **plus one false positive** | 0.80 | 1.00 | **0.8333** |

**Missing a true match scores better than adding a wrong one.** A threshold
chosen by pairwise F1 or accuracy would sit near 0.5; optimising macro F_0.5
directly places it at **0.60**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.890575 on the public leaderboard.**
  Held-out validation (US + India, full target pools) gave **0.9099** against an
  all-empty baseline of 0.0582. Validation was therefore ~2 points optimistic —
  well calibrated for a metric this sensitive.

  | reference | score |
  |---|---|
  | all-empty baseline | 0.0582 |
  | local validation (US + India) | 0.9099 |
  | **public leaderboard (US + India + France)** | **0.890575** |

  Blocking recall ceiling, measured against the full target pool: **US 0.9462,
  India 0.8796**. Pairwise average precision 0.9878.

  **The validation-to-leaderboard gap decomposes to France.** Validation could
  only cover US and India, because France appears nowhere in the training data.
  Taking the trained countries at their validation level:

  ```
  0.85 × 0.91 + 0.15 × France = 0.8906   =>   France ≈ 0.78
  ```

  France scores roughly 13 points below the trained countries on 15% of the
  test set. This is inference rather than direct measurement — it assumes
  US/India transfer cleanly — but an independent signal agrees: France abstains
  on 16.11% of entities while US abstains on 6.54% and the true singleton rate
  is 5.58%. Under F_0.5 abstention has no protective value on an entity that
  does have matches (it scores 0.0, exactly as a wrong guess would), so
  over-abstention is pure loss.

  The score exceeding the pair-level ceiling is expected, not an error: the
  ceiling is measured over *pairs* while the score is a *per-entity* macro
  average, in which an entity recovering 3 of 4 matches still scores 0.9375 and
  a correctly-empty singleton scores 1.0.

- **Common false positives (wrong merges):** chain businesses and franchises
  sharing a name within a city, distinguished only by street number; and
  generic short names where the normalised key collapses to one or two common
  tokens.

- **Common false negatives (missed matches):** dominated by blocking, not by
  the classifier. Average precision is 0.9878, so the scorer is close to
  saturated; recall is the binding constraint.

  Per-country recall ceilings make the cause explicit:

  | | India | US |
  |---|---|---|
  | blocking recall | **0.8796** | 0.9462 |
  | name non-Latin | 23.5% | 7.5% |
  | **address non-Latin** | **22.6%** | **0.0%** |
  | no Latin bridge at all | 5.6% | 0.0% |

  For India, character n-grams fail on *both* fields simultaneously. We had
  initially assumed the address would always serve as the bridge for
  cross-script names; that holds for US, where addresses are 100% Latin, and
  fails for India.

  Notably, India misses 11.5% of pairs while only 5.6% lack any cross-script
  bridge — so roughly **6% are ranking failures**, where the true match is
  representable but falls outside the top-40 candidates, rather than signal
  failures. That points at candidate depth rather than new signals.

  DBA/trade-name pairs are the other structural group: a legitimate match whose
  two names share almost no tokens cannot be recovered by string similarity.

### Hypotheses that did not survive measurement

We report these because they shaped the final design as much as the successes
did, and because each looked mechanically sound beforehand.

**Mutual exclusivity.** We verified that Source-2/3 records are mutually
exclusive across Source-1 entities (zero violations in 7,638,365 pairs) and
expected that resolving contested claims to the highest-scoring entity would be
a free precision gain. Measured, it changed the score by **0.0000** — at the
chosen threshold only about 1 claim in 15,000 is contested, because the
classifier is already precise enough that conflicts are vanishingly rare.
Retained as a cheap correctness guarantee; not claimed as a contribution.

**Numeric-token blocking.** Since digits are script-invariant, and 82.6% of
India's true pairs share an exact address number (19.7% having an unmatchable
non-Latin name *and* a shared number), an exact numeric index looked like the
natural fix for the India recall gap. Measured on India against the full
4,133,346-record target pool: recall went 0.8851 → **0.8893**, i.e. **+0.4
points for 34% more candidates.** The signal is largely redundant with address
blocking, because the address blocking key already contains those digits as
text and character 3-grams match them. The 82.6% figure was a correlation, not
untapped signal. Not enabled.

**Singleton detection as a primary lever.** Early analysis over-weighted this.
The measured singleton rate is **5.58%**, so an all-empty submission scores
0.0582 and the abstain decision is a guard rather than the main lever. The
dominant term is non-singleton set quality.

---

## 6. Conclusion

Treating candidate generation as the binding constraint, and measuring it
honestly against the full target pool rather than a convenient sample, mattered
more than model capacity: pairwise average precision reached 0.992 while total
recall remained capped at 0.93. The two decisions that moved the score most
were unioning name and address blocking to bridge cross-script records, and
tuning the decision threshold against macro F_0.5 itself. The main lesson was
methodological — three of our early conclusions (that singletons were a major
lever, that mutual exclusivity would pay, that a sampled target pool was
representative) were overturned by measurement, and each would have cost us had
we built on it.

---

## Appendix

### A. Code Artefacts

Complete pipeline in `code/business_entity_resolution/`, all source under
`src/`, with `README.md` giving exact reproduction commands and a pinned
`requirements.txt`.

| Module | Responsibility |
|---|---|
| `io_tsv.py` | TSV I/O: tab separator, no quoting, ids as strings, empty ≠ null |
| `normalize.py` | Vectorised normalisation and blocking-key construction |
| `blocking.py` | TF-IDF top-k candidate generation, union, recall-ceiling measurement |
| `features.py` | The 20 pairwise features |
| `model.py` | LightGBM training with entity-grouped splits |
| `decide.py` | Threshold, singleton gate, exclusivity resolution |
| `pipeline.py` | Orchestration and CLI |
| `submit.py` | Output writers enforcing every validator rule at write time |
| `metric.py` | Macro F_0.5 with the singleton convention |

Entry points:

```bash
python train_model.py --sample-entities 150000     # -> artifacts/model.txt, threshold.json
python -m pipeline predict --model ../artifacts/model.txt --threshold 0.60
```

`tests/test_metric.py` verifies the scorer against the worked example in the
problem statement (0.714) and pins the asymmetry values quoted in §4.

**Note for reviewers on macOS:** LightGBM requires the OpenMP runtime
(`brew install libomp`) or it fails at import.

### B. Additional Results

**Blocking parameter sweep** (20,000 Source-1 entities × 6,186,873 targets):

| `max_df` | n-gram | time | recall |
|---|---|---|---|
| none | (2,3) | 338 s | 0.9883* |
| 0.10 | (2,3) | 149 s | 0.9874* |
| **0.01** | **(3,3)** | **16 s** | **0.9709*** |
| 0.003 | (3,3) | 9 s | 0.8368* |

\* measured against a 369k sampled pool; see §3 on why these over-state recall.
Against the full pool the chosen configuration yields **0.9278**.

**Decision threshold curve** (macro F_0.5, held-out entities):

| threshold | 0.05 | 0.25 | 0.45 | 0.55 | **0.60** | 0.65 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| macro F_0.5 | 0.821 | 0.907 | 0.924 | 0.930 | **0.931** | 0.931 | 0.930 | 0.917 |

The curve is flat near its peak, so the chosen threshold is not finely tuned to
the validation split.

**Feature importance** (LightGBM gain, %):
`nums_token_set` 33.6 · `name_ratio` 20.3 · `addr_cos` 10.0 ·
`addr_token_set` 9.7 · `addr_token_sort` 7.8 · `name_token_sort` 5.5 ·
`name_jaro` 4.3 · `name_token_set` 2.8

Notably, `nums_exact` has the strongest raw class separation of any single
feature (0.615) yet contributes only 0.9% of gain — `nums_token_set` subsumes it
and is more granular.
