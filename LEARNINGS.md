# LEARNINGS — every observation, in detail

Running record of what was measured, what was believed and disproved, and what
it cost. Kept because six confident predictions were overturned by measurement
in a single session, and the pattern matters more than any individual result.

Companions: `CONTEXT.md` (state + facts), `ps.md` (problem statement),
`brainstorm.md` (strategy), `Documentation_template.md` (submission write-up).

---

## 0. WHERE WE STAND

| | |
|---|---|
| Our leaderboard score | **0.890575** |
| **Top of leaderboard** | **0.985884** |
| Top 500 cutoff | **>0.95** |
| **Gap to close** | **~0.06 to reach top-500, ~0.095 to win** |

We are well outside the top 500. That is not a tuning gap — something
structural is missing.

---

## 1. THE CENTRAL DIAGNOSIS

**Blocking recall is the ceiling, and ours is far too low.**

| | value |
|---|---|
| Pairwise classifier AP | **0.9878** — near saturated, NOT the bottleneck |
| Blocking recall US | 0.9462 |
| Blocking recall India | 0.8796 |

A pair never proposed can never be scored. If 500 teams exceed 0.95 macro
F₀.₅, their recall is likely ~0.98. Ours at ~0.93 explains the entire gap
without needing any other hypothesis.

**Corollary that must not be forgotten: changing the MODEL cannot help.** At
AP 0.9878 a transformer, a different booster, or an ensemble moves almost
nothing. Every hour on model architecture is an hour not spent on recall.

---

## 2. MY LIKELY ROOT-CAUSE ERROR: the max_df trade

`max_df=0.01` was chosen by measuring on a **369k sampled target pool**:

| config | toy pool recall | time |
|---|---|---|
| no max_df | 0.9883 | 338 s |
| max_df=0.01, 3-gram | 0.9709 | 16 s |

A 1.7-point loss for a 21x speedup looked obviously correct. **It was never
re-validated against the full 6.19M pool.** At real scale that setting leaves:

- vocabulary: only **24,824** n-grams
- **7.5 trigrams retained per name** — a 25-char name has ~25, so ~70% of the
  signal is discarded
- measured real-pool recall: **0.9278**

The toy pool almost certainly understated the cost. Being measured now.

**Lesson: a hyperparameter tuned on a sample must be re-validated at full
scale before it is locked. Sampling changes the difficulty of the task, not
just its size.**

---

## 3. PREDICTIONS I GOT WRONG (7)

| # | Predicted | Measured | Cost |
|---|---|---|---|
| 1 | Singletons a major lever (~35%) | **5.58%** | Misallocated early design effort |
| 2 | Mutual exclusivity buys precision | **0.0000 gain** (1 conflict in 15,000) | Wasted implementation |
| 3 | Blocking recall 0.9709 | **0.9278** on the real pool | Over-stated our position by 4 pts |
| 4 | Numeric-token blocking closes the India gap | **+0.004** for +34% candidates | Wasted a build + measurement cycle |
| 5 | "Address is the bridge for cross-script" | True for US (0% non-Latin addrs), **false for India (22.6%)** | Wrong mental model of the hardest partition |
| 6 | Runtime estimates (x4) | France 30→13 min, US 2.5→2.05 h, India 4→3.7 h | Repeated mis-planning |
| 7 | max_df trade is cheap | under measurement — suspected primary cause of the gap | Possibly the whole 0.06 |

**The pattern: a plausible mechanism, asserted confidently, contradicted by
measurement.** Every single one was caught only by measuring. The process
works; the instinct to trust the mechanism does not.

---

## 4. BUGS I INTRODUCED AND FIXED

1. **Index-offset mismatch** — offset predictions but not ground truth ⇒ the
   second country scored 0 on every entity ⇒ bogus 0.4690. Now pinned by
   `test_offset_predictions_against_unoffset_truth_scores_zero`.
2. **SciPy prunes explicit zeros** in sparse addition, so `zeros + X` collapses
   to X's own pattern ⇒ misaligned candidate arrays. Fixed with a sorted
   int64-key gather plus a length assertion.
3. **Ran blocking single-threaded for hours** — `sparse_dot_topn` does
   `n_threads or 1`, so 0/None is SERIAL. Cost 2.7x.
4. `.gitignore` `artifacts/**` did not match at depth ⇒ a 4 MB model nearly
   committed.
5. **Poller treated an auth failure as job completion** ⇒ falsely reported "all
   finished" after 81 min when nothing had finished.
6. **Renamed a Polars kwarg** to silence a local deprecation warning ⇒ broke
   the Kaggle run (older Polars). Now version-detected at import.
7. Large Kaggle outputs **download partially with no error** ⇒ verify by size.

---

## 5. WHAT IS MEASURED AND SETTLED

### Data (all measured, full files)
- train S1 2,206,821 / S2 5,034,616 / S3 5,285,603
- test S1 1,732,544 / S2 4,887,273 / S3 5,082,316
- **Singletons 5.58%** ⇒ all-empty scores 0.0582
- Mean 3.67 matches, max 11; 7,638,365 true pairs
- **Zero cross-country matches** (693,069 checked) ⇒ lossless partition
- **Perfect mutual exclusivity** — 0 of 7.6M ids claimed by 2 entities

### Cross-script (the hard part)
| | India | US |
|---|---|---|
| name non-Latin | 23.5% | 7.5% |
| **address non-Latin** | **22.6%** | **0.0%** |
| no Latin bridge at all | 5.6% | 0.0% |

India is 47% of the test set and has the worse recall. US addresses being
100% Latin is why "address is the bridge" held there and failed here.

### Settled parameters
| param | value | evidence |
|---|---|---|
| ngram | (3,3) char_wb | faster AND better recall than (2,3) at max_df=0.01 |
| min_sim | inert | 0.25 vs 0.10 gave identical recall |
| n_threads | **-1** | 0/None is serial |
| threshold | 0.575 | swept on macro F₀.₅; curve flat 0.475-0.65 |
| top_n | 40 (under review) | 0.9279@20 → 0.9397@30 → 0.9462@40; **blocking time FLAT in top_n** |

---

## 6. HYPOTHESES FOR THE REMAINING GAP (ranked)

**H1 — max_df destroyed recall at scale.** Under measurement. Blocking is
~2.6 h for the full test, so even a 5x slowdown is affordable.

**H2 — top_n far too low.** Blocking time is flat in top_n; only downstream
scoring grows. 150-300 candidates is cheap.

**H3 — the noise is GENERATED, not natural.** Evidence: `"-- Holloway Peak Inc
Seafood"`, systematic legal-suffix swaps, controlled typos, token
transpositions, Devanagari transliterations. If the organisers synthesised the
corruption with a fixed transform set, then **explicitly inverting each
transform** beats general fuzzy matching by a wide margin. A 0.986 top score is
implausibly high for genuinely messy real-world ER — this is the theory that
best explains the top of the leaderboard.

**H4 — transliteration.** 23.5% of India names are non-Latin. Rejected earlier
using the GLOBAL 2.26% no-bridge figure when India's is 5.6% — wrong statistic
on the partition that is 47% of the test set.

**H5 — multilingual embeddings** (LaBSE / multilingual-E5; MIT/Apache, <8B, so
licence-clean). The only approach that addresses DBA/trade names, where two
legitimate names share almost no characters. Kaggle provides a free T4.

---

## 7. OPERATIONAL (cost ~5 failed runs)

- **Local M1 (8 GB) resets under load** — memory, not heat: two countries'
  TF-IDF + both feature matrices + a `np.vstack` duplicate ≈ 6-7 GB
- **AWS unusable** — account on the Free Plan; largest allowed instance
  `m7i-flex.large` (2 vCPU/8 GiB) is *smaller than the laptop*. Needs a Paid
  Plan upgrade. If upgraded: `c6a.16xlarge` 64 vCPU/128 GiB @ $1.496/h is best
  value (half the Intel price)
- **Kaggle: 4 vCPU, 31 GB RAM**, free, 12 h sessions
- `enable_internet: true` is **ignored** unless the account is phone-verified
- **`kernel_sources` silently does not mount** → publish artifacts as a DATASET
- OAuth expires **~3 h**, CLI does not auto-refresh → `kaggle auth login --force`
- Kaggle Polars is **1.35.2** (local 1.44) → version-agnostic kwargs required
- macOS has no `timeout` (it is `gtimeout`)

### Runtimes (Kaggle, 4 vCPU)
| stage | entities | candidates | time |
|---|---|---|---|
| train | 100k/country | — | 71 min |
| France | 259,452 | 19.5M | 13 min |
| US | 663,106 | 50.7M | 123 min |
| India | 809,986 | 62.9M | 223 min |

---

## 8. FRANCE — the unvalidatable partition

France is 15% of test and appears **nowhere** in training.

| country | abstention | in train? |
|---|---|---|
| US | 6.54% | yes |
| India | 8.58% | yes |
| **France** | **16.11%** | **no** |
| *(true singleton rate)* | *5.58%* | |

US and India sit near the true rate; France abstains ~3x as often. Estimated
from the leaderboard decomposition: **France scores ~0.78 vs ~0.91**, a 13-point
gap on 15% of the test set.

Under F₀.₅, abstaining on an entity that HAS matches scores 0.0 — identical to
guessing wrong. Abstention has no protective value except on true singletons,
so over-abstention is pure loss.
