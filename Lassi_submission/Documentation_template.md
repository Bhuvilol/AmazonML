# ML Challenge 2026 — Business Entity Resolution
## Team Lassi — Methodology Document

**Final leaderboard score: 0.937** (macro-averaged F₀.₅)
**Progression: 0.890575 → 0.893467 → 0.937**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Analysis](#2-problem-analysis)
3. [Development Methodology](#3-development-methodology)
4. [Data Measurement](#4-data-measurement)
5. [Normalisation](#5-normalisation)
6. [Candidate Generation (Blocking)](#6-candidate-generation-blocking)
7. [Feature Engineering](#7-feature-engineering)
8. [Matching Model](#8-matching-model)
9. [Decision Stage](#9-decision-stage)
10. [France and Open-Set Handling](#10-france-and-open-set-handling)
11. [Experimental Log](#11-experimental-log)
12. [Results and Error Analysis](#12-results-and-error-analysis)
13. [Engineering Defects Found and Fixed](#13-engineering-defects-found-and-fixed)
14. [Heuristics Discovered](#14-heuristics-discovered)
15. [Reproducibility](#15-reproducibility)
16. [Output Validation](#16-output-validation)
17. [What We Would Do Next](#17-what-we-would-do-next)

---

## 1. Executive Summary

We built a three-stage entity resolution pipeline — **recall-greedy candidate
generation → precision-calibrated pairwise scoring → per-entity set decision**
— and improved it from 0.8906 to **0.937** through seven controlled
experiments, four of which we rejected on their own measurements.

The shipped system is a **43-feature LightGBM** pairwise classifier over a
candidate set built from four complementary blocking views, with a single
global decision threshold and mutual-exclusivity resolution.

### 1.1 What moved the score, and what did not

| Intervention | Measured effect | Shipped? |
|---|---|---|
| Union of name **and** address blocking | +14% recall vs name-only | ✅ |
| Transliteration (Indic → Latin) | 34.9% of cross-script pairs made reachable | ✅ |
| `max_df` 0.01 → 0.50 | +2.9 US / +2.6 India pts recall ceiling | ✅ |
| Exact-key blocking (hash join) | +0.7 US / +2.3 India pts recall ceiling | ✅ |
| **IDF / rare-token features** | **+0.0097 US / +0.0223 India** (locked holdout) | ✅ |
| **Competition (rank/share) features** | **+0.0053 US / +0.0068 India** (locked holdout) | ✅ |
| Mutual-exclusivity resolution | 3 / 16 assignments changed — near-null but free | ✅ |
| Cardinality-aware decision policies | −0.0005 to −0.0033 across 236 variants | ❌ |
| Learned normalization (mined rewrites) | 9 India / 2 US blocked pairs recovered | ❌ |
| Reverse target→S1 retrieval | +1.3% recall at 25–414 candidates per pair | ❌ |
| Numeric-token blocking | +0.7% recall at 708 candidates per pair | ❌ |
| Corruption inversion | +0.061 cosine but +0.0024 recall@40 | ❌ |
| Per-entity expected-F optimiser | **−0.0131** | ❌ |
| Learned cardinality selector | **−0.0060** | ❌ |

### 1.2 The single most important methodological decision

**Every idea was measured against a locked holdout before adoption.** Seven of
our fourteen candidate improvements were harmful, inert, or uneconomic. At
least three of them — the expected-F optimiser, the cardinality policies, and
reverse retrieval — we would have shipped on intuition alone.

The pipeline's final gain (+0.0435 on the leaderboard) came from two feature
families that took roughly four hours to implement. The preceding thirty hours
were spent on measurements that told us what *not* to build.

---

## 2. Problem Analysis

### 2.1 The metric determines the architecture

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

macro-averaged **per Source-1 entity**, with singletons scoring 1.0 for a
correct empty prediction and 0.0 for any prediction.

This single formula drove nearly every design decision. Worked through for an
entity with 4 true matches:

| Prediction | P | R | F₀.₅ |
|---|---|---|---|
| all 4 correct | 1.00 | 1.00 | **1.0000** |
| 3 of 4, no false positive | 1.00 | 0.75 | **0.9375** |
| all 4 **plus one false positive** | 0.80 | 1.00 | **0.8333** |

**Missing a true match scores strictly better than adding a wrong one.** This
is not a tiebreak — it is the dominant gradient of the problem.

Verified as an assertion, not an assumption, in `tests/test_metric.py`:

```python
def test_missing_a_match_beats_adding_a_false_positive():
    truth = {"a", "b", "c", "d"}
    miss_one        = entity_f_beta({"a","b","c"}, truth)            # 0.9375
    all_plus_one_fp = entity_f_beta({"a","b","c","d","x"}, truth)    # 0.8333
    assert miss_one > all_plus_one_fp
```

### 2.2 The consequence: opposite biases in adjacent stages

Because precision is purchasable downstream but recall is not, the two stages
must be biased **oppositely**:

- **Candidate generation must be recall-greedy.** A true pair not proposed here
  can never be recovered, no matter how good the model is.
- **The decision stage must be precision-biased.** This is where the F₀.₅
  asymmetry is paid for.

We believe conflating these is the most common way to lose score on this
problem. Blocking tightly "to keep precision high" discards recall permanently
and gains nothing, because precision is bought at the threshold.

### 2.3 The trivial floor

Predicting empty for every entity scores exactly the singleton rate. We
measured **5.58%** (123,247 of 2,206,821 training entities). This reframes every
false positive on a singleton as clawing back a point already held.

### 2.4 Correction to an early framing

Our initial instinct was "maximise accuracy." That is actively misleading here:
after blocking, the candidate set is **96.3% negative**, so a classifier
predicting "no match" everywhere achieves 96.3% pairwise accuracy and scores
0.0558. Pairwise accuracy does not translate monotonically into macro F₀.₅,
because the metric evaluates the *composition of each entity's predicted set*,
not the average correctness of individual pair decisions.

We therefore implemented the exact competition metric first — before any model
code — and unit-tested it against the problem statement's worked example
(0.714) plus the four cases involving a 0/0 division. Everything downstream is
calibrated against that module, so it had to be right before anything else
existed. 28 tests pin it.

---

## 3. Development Methodology

### 3.1 Loop structure with numeric exit criteria

We worked in loops, each terminating on a **measurement**, not a feeling:

| Loop | Goal | Exit criterion |
|---|---|---|
| 0 | Metric, I/O, submission writer, synthetic data | Metric returns 0.714 on the PS example; validator PASSes on synthetic output |
| 1 | Measure the data (§4) | Eight empirical questions answered in writing |
| 2 | Establish the floor | All-empty submission scores exactly the singleton rate |
| 3 | Blocking recall ceiling | Ceiling plateaus; chosen `top_n` documented with the curve |
| 4 | Pairwise scorer | Stable entity-grouped CV; calibrated probabilities |
| 5 | Decision layer | macro F₀.₅ maximised; any step not paying for itself removed |
| 6 | Generalisation hardening | Cross-country gap understood and documented |
| 7 | Package and document | Validator PASS; clean clone regenerates both outputs |

### 3.2 The locked-holdout protocol

From Experiment 1 onward, every comparison used the same discipline:

- **15,000 held-out Source-1 entities per country**, fixed seed 777, blocked
  against the **full** country target pool.
- **Training entities disjoint from evaluation entities** (seed 1234, sampled
  from the remainder), asserted at runtime.
- The holdout is split **50/50 into A and B** (rng seed 5). Thresholds and
  hyperparameters are selected on **A**; the reported number is **B**, untouched.
- Both arms of any A/B train on the same rows and score the same rows. The only
  difference is the feature matrix.

This protocol exists because of a defect we found in our own earlier work: the
original `train_model.py` selected the threshold by sweeping macro F₀.₅ on the
validation entities and then **reported that same maximum as the validation
score** — an optimistically biased estimate. We quantified the bias directly:

| | selection bias (A − B) |
|---|---|
| India | −0.0037 |
| US | −0.0001 |

Small, but only knowable by measuring it. Split B actually scored *higher* than
A, which also told us our earlier local numbers were not inflated by threshold
selection — removing one candidate explanation for the local-to-leaderboard gap.

### 3.3 Rejection is a first-class outcome

A step that does not beat its predecessor by more than fold noise is
**reverted, not kept**. This rule killed the expected-F optimiser (−0.0131), the
learned cardinality selector (−0.0060), and all 236 cardinality-policy variants.

### 3.4 Evidence tiers

Throughout development we tagged every claim:

- `[M]` measured on real data
- `[L]` leaderboard-confirmed
- `[C]` read from code
- `[P]` pending measurement

This prevented inferences from hardening into facts. One example of why it
mattered: for several hours we prioritised work on France based on the figure
"France ≈ 0.78", which was **arithmetic**, not measurement —
`0.85 × 0.91 + 0.15 × France = 0.8906`, resting on the assumption that US and
India scored their *local validation* figure on the *test* set. Direct
measurement later showed France's blocking was excellent (99.8% of entities had
a top-1 address cosine above 0.5), and the premise was wrong.

---

## 4. Data Measurement

Answered before any modelling work. All figures measured, not estimated.

### 4.1 Scale

| File | Rows | By country |
|---|---|---|
| `train_source1.tsv` | 2,206,821 | US 1,323,633 · India 883,188 |
| `train_source2.tsv` | 5,034,616 | US 3,016,817 · India 2,017,799 |
| `train_source3.tsv` | 5,285,603 | US 3,170,056 · India 2,115,547 |
| `test_source1.tsv` | 1,732,544 | US 663,106 · India 809,986 · **France 259,452** |
| `test_source2.tsv` | 4,887,273 | US 1,871,330 · India 2,312,565 · France 703,378 |
| `test_source3.tsv` | 5,082,316 | US 1,945,701 · India 2,405,000 · France 731,615 |

### 4.2 Ground truth structure

| Statistic | Value |
|---|---|
| S1 entities with a ground-truth row | 2,206,821 (**all** of them) |
| Singletons (zero matches) | **123,247 = 5.58%** |
| Total matched IDs | 7,638,365 |
| Split by source | S2 3,693,619 (48.4%) · S3 3,944,746 (51.6%) |
| Mean matches per S1 | 3.461 |
| Mean over non-singletons | **3.666** |
| Maximum cardinality | **11** |

Cardinality distribution:

| matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count (k) | 123 | 119 | 375 | **531** | 484 | 322 | 165 | 64 | 19 | 4.2 | 0.5 | 0.04 |
| share | 5.58% | 5.40% | 17.00% | **24.05%** | 21.94% | 14.59% | 7.47% | 2.90% | 0.85% | 0.19% | 0.02% | 0.00% |

The modal entity has **3** matches, and 74% have between 2 and 5.

### 4.3 Three structural findings that shaped the architecture

**(a) Mutual exclusivity is perfect.** Of 7,638,365 distinct S2/S3 IDs used in
ground truth, **exactly 0 are claimed by more than one Source-1 entity.** This
is a hard global constraint and we exploit it in the decision stage.

**(b) Matches never cross countries.** Verified on all training pairs. Country
is therefore a **lossless hard partition**, which reduces the candidate space
by orders of magnitude at zero recall cost. We did not assume this — we
checked it, because if cross-country matches existed even rarely, a hard
partition would put them permanently beyond the recall ceiling.

**(c) 26% of targets are pure distractors.** 7,638,365 of 10,320,219 S2+S3
training records are matched to something (74.0%); the remaining **2,681,854
match nothing at all**. The candidate generator must therefore discriminate
against a large deliberately-unmatched population, not merely rank.

### 4.4 Field quality

| | S1 | S2 | S3 |
|---|---|---|---|
| Name missing | 0.000% | 0.000% | 0.000% |
| Address missing | 0.000% | **3.356%** | **3.328%** |
| Country missing | 0 | 0 | 0 |
| Exact duplicate (name\|address) | **0.00%** | 0.51% | 0.36% |
| `entity_id` unique | ✅ | ✅ | ✅ |

Source 1 being 0.00% duplicated confirms the problem statement's description of
it as the deduplicated reference source.

### 4.5 Script distribution — the finding that determined blocking

Measured across all 7,638,365 true training pairs:

| Source-2/3 name | Source-2/3 address | share of true pairs |
|---|---|---|
| Latin | Latin | 79.30% |
| Latin | non-Latin | 6.81% |
| **non-Latin** | **Latin** | **11.63%** |
| non-Latin | non-Latin | 2.26% |

Source 1 is **100% ASCII**; Sources 2 and 3 are 11–15% non-Latin (Devanagari,
Tamil, Telugu, Gujarati, Odia). For the 11.63% with a non-Latin name but a
Latin address, **name similarity is structurally zero** — the two strings share
no characters at all — and only the address can propose the pair.

**This single table is why the candidate set is a union rather than an
intersection, and why name-only blocking concedes roughly 14% of recall.**

---

## 5. Normalisation

`src/normalize.py` → `add_normalized_columns()`. Applied identically to all
three sources, producing seven derived columns consumed by blocking and
features.

### 5.1 Pipeline

```
raw → transliterate → fold accents → strip punctuation
    → normalize_name / normalize_address       (comparison forms)
    → blocking_name / blocking_address         (blocking keys)
    → numeric_tokens, non-ASCII flags
```

**Step 1 — Transliteration** (`unidecode`). Devanagari/Tamil/Telugu/Gujarati/
Odia → Latin. Measured on 4,000 true pairs with a non-Latin target name:

| key | mean similarity | median | fraction > 0.15 |
|---|---|---|---|
| raw | 0.286 | 0.000 | 43.1% |
| **transliterated** | **0.479** | **0.267** | **73.4%** |

**34.9% of cross-script pairs move from effectively zero similarity to
usable.** Without this step those pairs are invisible to character n-grams no
matter how the ranking is tuned. This is one of the highest-leverage
interventions in the entire pipeline and it costs a single library call.

**Step 2 — Accent folding.** A vectorised Latin-diacritic map (`á→a`, `ñ→n`,
`ß→ss`, `æ→ae`, …) rather than `unicodedata` NFKD. The map is vectorisable in
Polars; NFKD would force a per-row Python call across ~12M records.

**Step 3 — Punctuation.** `&` → `and` (the PS names this explicitly),
apostrophes deleted rather than spaced (`O'Brien` → `obrien`, not `o brien`),
all other punctuation → space, whitespace collapsed.

**Step 4 — Address abbreviations.** A hand-written map applied token-wise:
`rd→road`, `ave→avenue`, `blvd→boulevard`, `ct→court`, `ln→lane`, `hwy→highway`,
`ste→suite`, `bldg→building`, `nr→near`, `opp→opposite`, `mkt→market`,
`rly→railway`, plus directionals. This is domain knowledge expressed in code,
not an external data lookup.

**Step 5 — Legal-suffix removal and token sorting** (blocking keys only).

### 5.2 Token sorting is deliberate

The problem statement promises **word-order transpositions** and **address
component reordering**. Sorting the tokens makes the blocking key invariant to
both:

```
'Sun Constructions Pvt. Ltd.'      → 'constructions sun'
'SUN CONSTRUCTIONS PRIVATE LIMITED'→ 'constructions sun'    ← identical
```

### 5.3 Legal suffixes, including empirically derived ones

The Latin list covers `inc`, `corp`, `llc`, `ltd`, `limited`, `pvt`, `private`,
`llp`, plus French `sarl`, `sas`, `sasu`, `sa`, `eurl`, `sci`, `snc` — the
French forms included deliberately because France is unseen and hand-written
suffix knowledge is allowed where external data lookup is not.

More interestingly, we **derived transliterated suffixes empirically from the
data** rather than guessing them:

```
limittedd, limitted, limittett, limirrrrdd, limitedd,
praaivett, praiveett, praaibhett, piraiveett, praivrrrr,
praaivet, praa, li, elelpii, prai
```

These are corrupted transliterations of "Limited" and "Private" that the Latin
suffix list does not recognise. Left in place they inflate the blocking key with
high-frequency, zero-information tokens.

### 5.4 Fallback behaviour

If suffix removal empties a name entirely (a business literally named
"Private Limited"), the key falls back to the unstripped form rather than
becoming an empty string that would collide with every other empty key.

---

## 6. Candidate Generation (Blocking)

The stage that sets the hard upper bound on recall. Every parameter was chosen
by measurement against the **full** target pool.

### 6.1 Shipped strategies

| # | Strategy | Key | Parameters |
|---|---|---|---|
| 1 | Name cosine top-k | char 3-gram TF-IDF, `char_wb` | `top_n=40`, `min_sim=0.25` |
| 2 | Address cosine top-k | char 3-gram TF-IDF, `char_wb` | `top_n=40`, `min_sim=0.30` |
| 3 | Exact name key | hash join on `name_block` | `max_group=100` (truncate) |
| 4 | Exact address key | hash join on `addr_block` | `max_group=100` (truncate) |

### 6.2 Vectoriser configuration

```python
TfidfVectorizer(
    analyzer="char_wb",      # word-boundary-aware character n-grams
    ngram_range=(3, 3),
    min_df=2,
    max_df=0.50,
    lowercase=False,         # normalize.py already lowercased
    dtype=np.float32,
)
```

**Why character n-grams rather than word tokens:** they are language- and
script-agnostic, degrade gracefully on diacritics, transliterations and typos
(all three named in the PS noise spec), and are order-insensitive by
construction. Word-token methods are none of these things.

**Why `char_wb`:** it respects word boundaries, so n-grams do not span across
token gaps and produce spurious matches on concatenations.

**Why fitted on the target side:** Sources 2/3 form the larger corpus and give
more stable IDF weights, and both sides must share one vocabulary for the dot
product to be a cosine similarity at all.

**Why phonetic keys were rejected:** Soundex and Metaphone encode English
phonology. They would mangle Indian and French names and actively harm the
open-set case. This was a deliberate exclusion, not an oversight.

### 6.3 Top-k similarity

`sparse_dot_topn.sp_matmul_topn`, chunked at 20,000 source rows. The routine
keeps only the largest `top_n` values per row **during** the multiply, so the
full |S1| × |S2∪S3| product is never materialised — at 1.7M × 10.3M that
product is ~17 trillion cells.

⚠️ **`n_threads` must never be 0 or `None`.** `sparse_dot_topn` evaluates
`n_threads or 1`, so both silently run **serial**. We lost hours to this before
noticing: a measured ~2.7× slowdown with no warning.

### 6.4 Exact-key blocking: converting a ranking into a guarantee

Top-k ranking has a structural failure mode no amount of parameter tuning
fixes. Our miss analysis found that **85.4% of US misses and 50.7% of India's**
were "Latin script, good token overlap → **ranking** failure", including pairs
whose normalised keys are **byte-identical**:

```
'Office of Housing'            vs  'Office Of Housing'         jaccard 1.00
'Great Media Private Limited'  vs  'Great Media Private Ltd'   jaccard 1.00
```

They are missed because common names have hundreds of key-mates and top-40
cannot hold them all — and `max_df` pruning strips exactly the common n-grams
such names are built from, so their similarity is computed on almost nothing.

A hash join has **no ranking cutoff**: if the keys match, the pair is proposed.
That is a guarantee rather than a ranking, which is precisely what the failure
mode calls for.

⚠️ **A subtle defect worth documenting in full.** Our first implementation
*skipped* key groups larger than `max_group=100`. Measured gain: **+0.67
points**. The skip removed exactly the common-name cases the blocker exists to
catch — the pathological groups *are* the target population. Changing "skip" to
"truncate to a bounded sample" raised the gain to **+2.3 points on India**.

*A cost cap intended to bound work can silently defeat the feature's entire
purpose. If a mechanism exists to handle a pathological case, the cap must not
exclude the pathology.*

### 6.5 Union assembly and a SciPy trap

The candidate set is the element-wise union of all four matrices, with each
strategy's similarity score preserved separately for downstream features.

⚠️ **The obvious implementation is silently wrong.** Adding a zero-valued
union-pattern matrix to align scores does **not** work, because SciPy prunes
explicit zeros during sparse addition — the result collapses back to the
addend's own sparsity pattern, producing misaligned score arrays that look
plausible and are wrong.

The correct approach, and what we ship: encode each `(row, col)` as a sorted
`int64` key (`row * n_targets + col`), gather by binary search, and assert
output length:

```python
union_keys = union_rows * stride + union.indices
positions  = np.searchsorted(union_keys, sub_keys)
out        = np.zeros(len(union_keys), dtype=DTYPE)
out[positions] = sub.data
```

### 6.6 Parameter sweep (full pool, 30,000 entities/country)

| config | US ceiling | India ceiling | blocking time |
|---|---|---|---|
| `top_n=40`, `max_df=0.01` | 0.9464 | 0.8863 | 316 s |
| `top_n=40`, `max_df=0.01` + exact-key | 0.9531 | 0.9093 | 376 s |
| `top_n=100` + exact + `max_df=0.50` | 0.9825 | 0.9486 | 1412 s |
| `top_n=40`, `max_df=0.50`, no exact | 0.9728 | — | 597 s |
| **shipped: `top_n=40` + exact + `max_df=0.50`** | **0.9757** | **0.9254** | ~600 s |

**`top_n` deliberately kept at 40.** Raising it to 100 buys ~0.7 further points
of ceiling but roughly doubles both compute and candidate-set size. The two
knobs act on **different axes**:

- `top_n` controls candidate **volume** — how many neighbours are kept per entity
- `max_df` controls candidate **quality** — which n-grams carry weight

Since the challenge ranks smaller candidate sets higher, spending on quality
rather than volume is the correct trade. Recognising that these were two axes
and not one knob was itself a necessary insight — our original sweep varied
them together, making the result confounded.

⚠️ **`max_df` is a parameter we initially got wrong, and the reason matters.**
We first set `0.01`, justified as "a 21× speedup for 1.7 points of recall" —
measured on a **369k sampled** target pool. Re-measured at full scale the trade
was **2.6 points**, not 1.7, and at `0.01` roughly 70% of each name's n-grams
were being discarded.

*A smaller pool has ~10× fewer distractors competing for the same top-k slots.
Any blocking parameter tuned on a sample must be re-validated at full scale.*

### 6.7 Per-strategy attribution (15,000 held-out entities/country)

Each strategy run **separately** against the full pool, then compared:

| strategy | India recall | S2 | S3 | India cands | unique¹ | US recall | US cands | unique¹ |
|---|---|---|---|---|---|---|---|---|
| name TF-IDF | 0.5676 | 0.5303 | 0.6030 | 600,000 | 1,497 | 0.7002 | 600,000 | 1,161 |
| **address TF-IDF** | **0.8097** | 0.8858 | 0.7373 | 599,932 | **13,907** | **0.8966** | 599,901 | **9,167** |
| exact name | 0.4445 | 0.4399 | 0.4489 | 360,413 | 197 | 0.5113 | 295,832 | 206 |
| exact address | 0.1024 | 0.1499 | 0.0571 | 6,286 | **0** | 0.1922 | 12,289 | **0** |
| numeric (not shipped) | 0.2689 | 0.2571 | 0.2801 | 273,909 | 358 | 0.1946 | 296,100 | 118 |
| **union (shipped)** | **0.9250** | 0.9407 | 0.9100 | 1,302,772 | — | **0.9763** | 1,363,769 | — |

¹ true pairs recovered by this strategy and by *no other shipped strategy*

Cumulative, in order:

```
India:  name 0.5676 → +addr 0.9212 → +exact_name 0.9250 → +exact_addr 0.9250
US:     name 0.7002 → +addr 0.9723 → +exact_name 0.9763 → +exact_addr 0.9763
```

**Two honest observations from this table:**

1. **Address blocking carries the system**, contributing 13,907 / 9,167 unique
   pairs against name's 1,497 / 1,161. This inverts the naive expectation that
   business *names* are the primary matching signal.
2. **`exact_addr` contributes exactly zero unique pairs** at `max_df=0.50`. It
   was worth +2.3 points at `max_df=0.01`; relaxing the pruning made it fully
   redundant. We retained it because removing it would alter a validated
   configuration for no measured benefit, but we report it here as **inert**.

### 6.8 Final candidate set characteristics

| country | mean/S1 | p50 | p95 | p99 | max | zero-candidate entities |
|---|---|---|---|---|---|---|
| India | 87.8 | 78 | 123 | 179 | 180 | **0** |
| US | 87.3 | 76 | 178 | 180 | 180 | **0** |
| France | 90.4 | 78 | 179 | 180 | 226 | **0** |

**Reduction ratio 0.99998** against the full pair space (India 0.99997899,
US 0.99998530). Not one of the 1,732,544 test entities received zero candidates.

---

## 7. Feature Engineering

43 features in three groups. Column order is fixed and identical between
training and inference; `model.py` records `ALL_FEATURE_NAMES` on the booster so
a mismatch raises rather than silently misaligning.

### 7.1 Group A — 20 pairwise-absolute features

Computed with `rapidfuzz.process.cpdist` (elementwise, multithreaded).

| # | Feature | Definition |
|---|---|---|
| 0 | `name_token_set` | `fuzz.token_set_ratio` / 100 — order-insensitive, robust to extra/missing tokens |
| 1 | `name_token_sort` | `fuzz.token_sort_ratio` / 100 — order-insensitive, sensitive to content |
| 2 | `name_ratio` | `fuzz.ratio` / 100 (Indel) — catches typos |
| 3 | `name_jaro` | `JaroWinkler.normalized_similarity` — strong on short strings and shared prefixes |
| 4 | `name_cos` | char-3gram TF-IDF cosine, reused from blocking |
| 5–9 | `addr_token_set`, `addr_token_sort`, `addr_ratio`, `addr_jaro`, `addr_cos` | the same five on the normalised address |
| 10 | `nums_token_set` | token-set ratio over address digit runs |
| 11 | `nums_exact` | sorted digit string identical (0/1) |
| 12 | `nums_both_present` | both sides have ≥1 digit run (0/1) |
| 13 | `name_len_ratio` | `min(len) / max(len)` |
| 14 | `addr_len_ratio` | same on address |
| 15 | `name_tok_diff` | \|token-count difference\| |
| 16 | `addr_tok_diff` | same on address |
| 17 | `tgt_is_s3` | target is Source 3 (0/1) |
| 18 | `tgt_name_non_ascii` | target name contains non-ASCII (0/1) |
| 19 | `tgt_addr_non_ascii` | target address contains non-ASCII (0/1) |

**Numeric tokens matter more than their count suggests.** Digits survive
transliteration unchanged, making them the most reliable cross-script signal.
Measured: **82.6% of India's true pairs share an exact address number**, and
19.7% have an unmatchable name yet a matching number.

**The regime flags (17–19) matter more than they look.** For a cross-script
pair every name feature is ≈0. Without a flag, the model reads that zero as
*disagreement*. With it, the model can learn *"ignore name similarity in this
regime, trust the address"* as a conditional rule.

### 7.2 Group B — 13 exact-match and IDF features

Validated at **+0.0094** (Experiment 1).

| Feature | Definition |
|---|---|
| `exact_name_norm` | normalised names identical and non-empty (0/1) |
| `exact_addr_norm` | normalised addresses identical and non-empty (0/1) |
| `exact_both_norm` | both of the above (0/1) |
| `exact_name_block` | blocking keys identical (suffix-stripped, token-sorted) |
| `exact_addr_block` | same on address |
| `name_jaccard` | \|shared tokens\| / \|union\| |
| `addr_jaccard` | same on address |
| `name_idf_jaccard` | Σ idf(shared) / Σ idf(union) — IDF-weighted Jaccard |
| `addr_idf_jaccard` | same on address |
| `name_idf_coverage` | Σ idf(shared) / Σ idf(query tokens) |
| `addr_idf_coverage` | same on address |
| `name_max_shared_idf` | highest IDF among shared tokens |
| `addr_max_shared_idf` | same on address |

Word-level IDF is computed as `log(N / (1 + df)) + 1` (sklearn's smoothed form)
over the **target corpus** — the same corpus and the same moment at which
blocking fits its character vectoriser, so train and serve agree by
construction.

**The question these answer that nothing else could:** *are the tokens two
records share rare or ubiquitous?* Two businesses sharing "Services Pvt Ltd" is
near-meaningless; two sharing "Zephyrion" is near-conclusive. No feature in
Group A can distinguish those cases.

### 7.3 Group C — 10 competition features

Validated at **+0.0053 US / +0.0068 India** (Experiment 2).

| Feature | Definition |
|---|---|
| `rank_name`, `rank_addr`, `rank_comb` | 0-based descending rank within this entity's candidate list |
| `marg_name`, `marg_addr`, `marg_comb` | gap to this entity's best (≤ 0) |
| `rel_name`, `rel_addr` | ratio to this entity's best (≤ 1) |
| `n_cands` | how crowded this entity's field is |
| `share_comb` | this pair's share of the entity's total similarity mass |

where `comb = name_cos + addr_cos`.

**Why these exist.** Every feature in Group A describes a pair **in isolation**.
None can express whether a candidate is *the best of forty* or *the
thirty-seventh of forty* — yet that is precisely what determines whether it is a
match. A 0.7 name cosine means something entirely different as the strongest
option in a weak field than as an also-ran behind several near-identical rivals.

They cost **nothing** to compute: every input was already present in the
`CandidateSet` arrays blocking produces.

### 7.4 Two features built, measured, and deliberately removed

`tgt_degree` (how many entities claim this target) and `tgt_margin` (gap to the
best claim on this target) were implemented and were part of an earlier
+0.0094 measurement.

**They are target-global**, so their denominator is the partition being
processed — 30,000 entities during training versus 160,000–660,000 in an
inference shard. That is a **train/serve scale skew**, not leakage: the feature
means something different at serve time than at fit time.

We ablated them and shipped the 10 S1-local features, which are scale-invariant
ratios. The ablation confirmed the entity-local features carry the gain on their
own.

*A feature whose semantics depend on batch size is a liability even when it
improves validation.*

### 7.5 Provenance verification

Because competition features aggregate across a candidate list, we verified
mechanically that they cannot encode anything they should not:

- **Signature check:** `competition_local(row, name_cos, addr_cos, n_entities)`
  — `col` is *not a parameter*, so the function is structurally incapable of
  seeing target-side information.
- **AST identifier walk:** every `Name` and `Attribute` node the function
  references, checked against a forbidden set (`col`, `labels`, `truth`,
  `predict`, split masks, target arrays). The function touches only its four
  arguments and numpy calls.
- **Empirical invariant:** `share_comb` must sum to 1.0 per entity — an
  empirical proof it is an entity-local share, not a global statistic.

⚠️ Our first version of the AST check used **substring matching** and flagged
`minlen`**`gt`**`h` as containing the forbidden token `gt`. It would have
aborted the run 40 minutes in. Caught by testing the guard itself before
launching. *Verification code needs verifying.*

### 7.6 Feature classes we considered and did not ship

| Class | Reason |
|---|---|
| Country agreement | Constant within a partition (we partition by country) — no information |
| One-hot country | Explicitly warned against by the PS; would not generalise to France |
| Phonetic keys (Soundex/Metaphone) | English-phonology specific; would harm India and France |
| Address component decomposition | Requires parsing conventions that differ per country; untested against the open-set requirement |
| Multilingual embeddings | Narrow target (DBA/trade names only); GPU-dependent; string features already dominate importance |

---

## 8. Matching Model

### 8.1 Algorithm and configuration

**LightGBM binary gradient-boosted trees.**

```python
objective          "binary"
metric             ["binary_logloss", "average_precision"]
learning_rate      0.08
num_leaves         63
min_data_in_leaf   200
feature_fraction   0.9
bagging_fraction   0.8      bagging_freq  1
lambda_l2          1.0
num_boost_round    600      early_stopping_rounds  50
num_threads        0        (0 = all cores, correct for LightGBM)
seed               42       deterministic  True
```

**Why GBDT and not something larger:** the inputs are 43 dense, bounded,
tabular similarity scores — precisely the regime where gradient boosting is
strongest. It trains in minutes on CPU, needs no pretrained weights (so the
MIT/Apache and ≤8B-parameter constraints are satisfied *by construction* rather
than by argument), and yields feature importances that feed directly into this
document.

Hyperparameters were deliberately **not** extensively searched. With 43
informative features and millions of rows, capacity is not the binding
constraint — threshold placement and feature quality are worth vastly more than
tree depth, and every hour spent on a hyperparameter grid is an hour not spent
measuring.

### 8.2 Training data construction

`train_model.py:build_training_partition`:

1. Sample **30,000 Source-1 entities per country** (seed 42).
2. Block each against the **full** country target pool (India 4,133,346;
   US 6,186,873) — not a sampled pool.
3. Every resulting candidate pair becomes a training row.
4. Label 1 iff the pair appears in ground truth.

Measured positive rate: **3.69% India / 3.70% US**. Roughly 2.6M training rows
per country.

**Sampling is by entity, never by pair.** The metric is macro-averaged per
entity and the decision stage reasons over an entity's whole candidate list, so
training must see realistic per-entity candidate distributions. Sampling pairs
would distort how many candidates each entity has and break the link between
training and the objective.

**Negatives are every non-positive candidate** — no downsampling, no negative
mining. We considered hard-negative mining and rejected it on evidence: the
residual error analysis (§11.5) found **0 of 400** nearest-miss false negatives
were outranked purely by impostors, so the population hard-negative mining
targets does not meaningfully exist here.

### 8.3 Validation split

`model.py:entity_group_split` — grouped by Source-1 entity, `valid_fraction=0.2`,
seed 42.

Splitting by *pair* would leak: two pairs from the same entity share a Source-1
record and often near-duplicate Source-2/3 records.

### 8.4 Model performance

Validation `average_precision` = **0.9821**. The pairwise ranking is
near-saturated; this number is what told us the remaining loss was not in
discrimination but in retrieval and in set construction.

Best iteration 599 of 600 — the round cap bound, not early stopping, so the
model had not converged. A longer schedule is a plausible small gain we did not
pursue.

### 8.5 One model for everything

A single model serves **both countries and both target sources**. Source enters
only as the binary `tgt_is_s3`. Country is never encoded anywhere (§10).

We considered per-country models given the measured divergence in error
structure (§12.4) but did not ship them: France has no training data, so a
per-country design would need a fallback path that is itself unvalidated.

### 8.6 Calibration

`model.py:calibration_report` compares predicted probability against observed
rate per decile. The decision stage compares scores to a threshold, so gross
miscalibration would place that threshold wrongly.

**Honest disclosure:** we implemented this diagnostic but do not apply a
calibration transform. Our own audit caught that an earlier version of this
module's docstring claimed predictions were "calibrated and checked" when the
function was never called — a documentation/reality mismatch we corrected. The
threshold is selected by direct macro-F₀.₅ sweep, which is robust to monotone
miscalibration, so we judged a calibration layer unnecessary rather than
beneficial.

---

## 9. Decision Stage

```python
keep = probability >= threshold          # 0.675
if singleton_gate: keep &= best[row] >= singleton_gate    # not used
resolve_exclusivity(rows, cols, probabilities)
```

### 9.1 Threshold selection

Swept directly against **macro F₀.₅ on held-out entities** — not pairwise F1,
not accuracy, not AUC. This matters: a threshold chosen by pairwise F1 would sit
near 0.5; optimising the actual competition metric places it at **0.675**.

The curve is a broad plateau:

| threshold | 0.55 | 0.60 | **0.65** | 0.70 | 0.75 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| macro F₀.₅ | 0.9205 | 0.9217 | **0.9218** | 0.9209 | 0.9196 | 0.9171 | 0.9022 |

All values between 0.55 and 0.725 lie within 0.002 of the maximum. This
flatness is itself informative — it says the global threshold is genuinely
exhausted as a lever, and it is why Experiment 4 tested *shape-aware* decision
rules rather than more threshold search.

### 9.2 Singleton handling

An entity emits an empty list when no candidate clears the threshold. We tested
a **separate singleton gate** (a stricter second bar on the entity's best
score, decoupling "does this entity match anything" from "which ones"):
measured **+0.0033**, inside selection noise across six policy finalists.
Rejected on the rule that a change must beat its predecessor by more than fold
noise.

### 9.3 Mutual-exclusivity resolution

Exploits the measured fact (§4.3a) that ground truth assigns each S2/S3 record
to at most one Source-1 entity. Where two entities claim the same target, the
higher-scoring claim wins.

**Measured effect: 16 of 43,806 assignments changed (India), 3 of 45,838 (US).**

We report this honestly as **near-null**. We initially suspected the 0.0000 gain
was a wiring fault — a constraint that holds perfectly across 7.6M pairs
producing no improvement seemed implausible — and re-verified it by running
`select()` twice, with and without exclusivity, on identical scores. The null
is real: the model almost never double-claims a target. Kept because it is
correct and free.

⚠️ **Known limitation.** Exclusivity resolves **within each shard**. Because
India is sharded 5 ways and US 3 ways for compute, a target claimed by entities
in *different* shards survives. On our v2 run this affected **34,210 excess
claims = 0.66% of emitted IDs**. It is not a rejection condition. Fixing it
correctly requires persisted pair scores to decide which entity keeps a
contested target — a capability (`--save-scores`) we added after the production
run had already begun.

---

## 10. France and Open-Set Handling

France appears only in test (259,452 entities, 1,434,993 targets) and never in
training. The problem statement warns explicitly against hard-coding, filtering
or one-hot encoding country.

### 10.1 Our approach: never encode country as a value

- **No country feature.** Not one-hot, not ordinal, not embedded.
- **Country is used only as a partition key**, which is lossless because we
  verified no true match crosses countries (§4.3b).
- **Every component is structurally language-agnostic:** character n-grams
  degrade gracefully across scripts, numeric tokens survive transliteration,
  token sorting is order-invariant, and the legal-suffix list includes French
  forms.
- **Phonetic keys deliberately excluded** (§6.2).

The one place a country-conditional feature would have been natural — a
"same country" boolean — is constant within a partition and therefore carries
no information.

### 10.2 Validation without labels

France cannot be validated directly. We used three substitutes:

1. **A synthetic unseen country.** `tests/make_synthetic.py` generates a country
   ("Zephyria") present only in the synthetic *test* split, with its own legal
   suffixes (`SARL`, `SA`, `SAS`). The end-to-end test therefore exercises the
   open-set path by construction, every run.
2. **Leave-one-country-out** as a generalisation proxy during development.
3. **Distribution comparison against the training countries** post-inference.

### 10.3 Measured France behaviour

Under v1, France abstained on **16.11%** of entities against a training
singleton rate of 5.58% — a 2.5× divergence that looked like a generalisation
failure.

We investigated directly, sampling 4,000 France test entities against 120,000
targets:

```
top-1 address cosine per France entity:
  mean 0.8948   median 0.9088   ≥0.5: 99.80%   ≥0.7: 97.08%   zero: 0.00%
```

**France blocking is excellent.** 99.8% of entities already have a strong top-1
candidate and none have zero. The abstention was not a retrieval failure.

We also mined French-specific normalisation gaps and found real ones —
`R.` → `Rue` affects ~25% of French Source-2 addresses, and region↔department
substitution (`Nouvelle-Aquitaine` ↔ `Gironde`) is invisible to character
n-grams. Applying them raised mean address cosine by only **+0.0070** and moved
12 entities of 4,000 across the 0.7 threshold. Not shipped.

Under the final model France emits at **essentially the same rate as the
training countries** — the open-set path behaves correctly.

---

## 11. Experimental Log

Seven controlled experiments, each with a stated intention, a protocol, and an
outcome. Four were rejected.

### 11.1 Forensic audit — establishing the baseline

**Intention:** before optimising anything, determine where score is actually
lost. Distinguish three ceilings: **retrieval** (fraction of true pairs present
in candidates), **model** (best macro F₀.₅ over the blocked set at any
threshold), and **decision** (what we actually ship).

**Protocol:** 15,000 held-out entities per country against the full pool, using
the then-current 20-feature model, pinned to the exact commit that trained it.

**Gates, checked before interpreting anything:**
- Reconciliation: `TP + FN_blocked + FN_model == total_true_pairs` — exact in
  both countries.
- Pin verification: `git rev-parse` matches; `booster.num_feature() == 20`.
- Sanity anchors: union recall reproduced 0.9757 US / 0.9254 India to within
  0.0006.

**Findings:**

| ceiling | India | US |
|---|---|---|
| Retrieval | 0.9250 | 0.9763 |
| Model (best threshold over blocked set) | 0.8955 | 0.9463 |
| Decision (shipped threshold) | 0.8947 | 0.9463 |

**Model ceiling ≈ decision output** — gap 0.0008 India, 0.0000 US. The threshold
was already optimal to three decimal places.

FN allocation as a percentage of **all** ground-truth pairs (not of FNs, so the
two are directly comparable):

| | blocked out | model-rejected |
|---|---|---|
| India | 7.50% | 11.61% |
| US | 2.37% | 9.86% |

**The model was losing more than blocking was**, 4.2× more on US. Combined with
a leaderboard-measured recall→score conversion of only **13.5%** (§12.2), this
redirected all subsequent work away from candidate generation.

### 11.2 Experiment 1 — exact-match and IDF features

**Intention.** The audit showed exact-address candidate pairs had a **22.9×
lift** over base match rate (0.847 vs 0.037) and exact-both a **27.1× lift**
(1.0000 — a perfect rule). Yet no feature expressed either. Separately, no
feature could distinguish a shared *rare* token from a shared *ubiquitous* one.

**Protocol.** 20-feature baseline vs 20+13. Identical candidate set, identical
15k/country eval entities (seed 777), disjoint 15k training entities (seed
1234), identical A→B threshold protocol, identical LightGBM configuration. The
only difference is 20 vs 33 columns.

**Result:**

| Metric | Baseline | +13 | Δ |
|---|---|---|---|
| India A F₀.₅ | 0.8953 | 0.9129 | +0.0175 |
| **India B F₀.₅** | 0.8975 | **0.9198** | **+0.0223** |
| US A F₀.₅ | 0.9469 | 0.9596 | +0.0127 |
| **US B F₀.₅** | 0.9470 | **0.9567** | **+0.0097** |
| India model-rejected FN % | 10.71% | 6.81% | −3.90% |
| US model-rejected FN % | 9.41% | 7.46% | −1.95% |
| India FP | 1,970 | 1,774 | **−196** |
| US FP | 646 | 467 | **−179** |
| India TP | 42,550 | 44,581 | +2,031 |
| US TP | 45,497 | 46,503 | +1,006 |

**Accepted.** B ≥ A in three of four cases — the gain survives and grows on the
untouched half. **Precision and recall improved simultaneously** (TP up, FP
down): a strict Pareto improvement with no hidden trade.

**The stated rationale was wrong, and the experiment succeeded anyway.**
Breaking the baseline's model-rejected positives down by exact-match class:

| | India | US |
|---|---|---|
| exact name only | 611 (10.96%) | 891 (18.36%) |
| exact address only | 114 (2.05%) | 174 (3.58%) |
| **exact both** | **0 (0.00%)** | **0 (0.00%)** |
| neither | 4,849 (86.99%) | 3,789 (78.06%) |

**Zero rejected positives had exact-both in either country.** The 27× lift was
real but described a population the model already resolved perfectly. Confirmed
by importance: *no exact-match feature appears in either top-20*.

**The gain came from the IDF features.** India: `addr_idf_jaccard` (gain
667,189) and `addr_idf_coverage` (412,355) rank first and second overall.

We retain the exact-match features because they were part of the validated
43-column space, but report them as inert.

### 11.3 Experiment 2 — competition-local features

**Intention.** Test whether relative-to-peers information helps, using the
33-feature model as baseline. Deliberately excluding `tgt_degree` and
`tgt_margin` because of their train/serve scale skew (§7.4).

**Protocol.** Identical to Experiment 1 in every respect except the feature
matrix (33 vs 43), plus mechanical provenance verification (§7.5).

**Result:**

| Metric | Exp1 33f | Exp2 43f | Δ |
|---|---|---|---|
| India A F₀.₅ | 0.9129 | 0.9225 | +0.0096 |
| **India B F₀.₅** | 0.9198 | **0.9266** | **+0.0068** |
| US A F₀.₅ | 0.9596 | 0.9641 | +0.0045 |
| **US B F₀.₅** | 0.9567 | **0.9620** | **+0.0053** |
| India FP | 1,774 | 1,252 | −522 |
| US FP | 467 | 547 | +80 |

**Causal check — the decisive result.** Evaluating Exp2 at **Exp1's own
threshold**, so no operating-point movement can explain the gain:

```
India:  Exp1 B 0.9198 @0.60   Exp2 B @SAME 0.60 = 0.9263  (+0.0065)
US:     Exp1 B 0.9567 @0.75   Exp2 B @SAME 0.75 = 0.9620  (+0.0053)
```

**96% of India's gain and 100% of US's survive at a fixed threshold.** US's two
arms selected the identical threshold, so its result contains no operating-point
confound at all.

**Recovery by rank** — of the positives Exp1 rejected, which does Exp2 recover?

| rank_comb | US baseline rejected | recovered | rate | newly lost | net |
|---|---|---|---|---|---|
| 0 | 166 | 33 | 0.199 | 27 | +6 |
| 1 | 287 | 102 | 0.355 | 35 | +67 |
| 2 | 451 | 186 | 0.412 | 25 | +161 |
| **3** | 528 | 265 | **0.502** | 22 | **+243** |
| 4–5 | 683 | 303 | 0.444 | 41 | +262 |
| 6–7 | 304 | 90 | 0.296 | 19 | +71 |
| 8–15 | 382 | 52 | 0.136 | 48 | +4 |
| 16+ | 1,047 | 98 | 0.094 | 129 | −31 |
| **TOTAL** | | **1,129** | | **346** | **+783** |

**Recovery peaks at ranks 2–5 and collapses beyond rank 8.** That is exactly the
mechanism signature: the features rescue candidates that are close-but-misordered
and do nothing for ones buried deep. Recovery exceeds loss 3.3× on US, 1.6× on
India — not churn.

**Accepted.**

**Importance in the final model:**

| India | gain | US | gain |
|---|---|---|---|
| **`addr_idf_jaccard`** | **975,400** | `nums_token_set` | 849,586 |
| `nums_token_set` | 185,630 | **`share_comb`** | **635,435** |
| **`share_comb`** | **159,612** | `name_ratio` | 163,700 |
| `addr_idf_coverage` | 102,232 | `name_jaro` | 138,976 |
| `name_token_set` | 78,737 | `addr_idf_coverage` | 116,463 |
| `name_jaccard` | 60,244 | `addr_token_set` | 47,013 |
| `rank_addr` | 55,281 | `name_idf_jaccard` | 38,254 |

On India an IDF feature ranks **first overall**, above every string-similarity
feature. On US `share_comb` ranks **second**. Both added families are
load-bearing.

### 11.4 Experiment 3 — residual failure diagnosis

**Intention.** With both feature families in, characterise what is *still*
failing before choosing the next intervention.

**Protocol.** Same 43-feature model, same locked setup. Profile three
populations on identical axes: accepted true positives (A), remaining
model-rejected true positives (B), and false positives (C).

**Result — the central finding:**

| India, p10/p50/p90 | A accepted TP | **B residual FN** | **C false positive** |
|---|---|---|---|
| `name_cos` | [0.0, **0.934**, 1.0] | [0.0, **0.000**, 1.0] | [0.0, **0.000**, 1.0] |
| `addr_cos` | [0.0, **0.909**, 1.0] | [0.0, **0.760**, 0.953] | [0.0, **0.865**, 0.992] |
| `addr_idf_jaccard` | [0.350, **0.824**, 1.0] | [0.0, **0.525**, 0.871] | [0.0, **0.729**, 0.998] |
| `rank_comb` | [0, **2**, 42] | [2, **23**, 58] | [1, **14**, 47] |
| `share_comb` | [.013, **.0262**, .0354] | [.0099, **.0140**, .0269] | [.0113, **.0148**, .0285] |
| frac `name_cos`≈0 | 0.368 | **0.625** | 0.658 |

**B and C are nearly indistinguishable on every axis.** India `share_comb`
.0140 vs .0148; name_cos≈0 rate 62.5% vs 65.8%. On the address features the
**false positives look *better* than the true matches being rejected**
(addr_idf_jaccard 0.729 vs 0.525).

The 43-feature representation cannot separate these two populations. That is the
finding, and it is why we stopped adding features of the same kind.

**Reject-rate by feature quintile** (monotone, 9–16× spread):

| quintile | India `share_comb` reject rate | India `addr_idf_jaccard` reject rate |
|---|---|---|
| q1 (lowest) | 0.175 | 0.186 |
| q2 | 0.094 | 0.088 |
| q3 | 0.054 | 0.039 |
| q4 | 0.018 | 0.019 |
| q5 (highest) | 0.011 | 0.021 |

Residual positives sit overwhelmingly in the **lowest** quintile of both — we
are near the limit of what these two representations can express.

**Nearest-miss classification (400 highest-scoring residual FNs):**

| | India | US |
|---|---|---|
| outranked only by other **TRUE** candidates | 161 (80.5%) | 178 (89.0%) |
| outranked by a mix | 39 (19.5%) | 22 (11.0%) |
| **outranked only by FALSE candidates** | **0 (0.0%)** | **0 (0.0%)** |

**Not one of 400 was outranked purely by impostors.** This is what ruled out
hard-negative mining: the competitors are *correct*.

### 11.5 Experiment 4 — cardinality-aware decision policies — REJECTED

**Intention.** Experiment 3 showed residual FNs were outranked by *true*
candidates on high-cardinality entities, suggesting the model finds the right
entity set and emits too few members. Test whether a shape-aware decision rule
beats a flat threshold.

**Protocol.** No retraining, no new features, identical candidate set. **236
parameterisations across 6 policy families**, each mapping a descending-sorted
score vector to a prefix length — a reduction that keeps every policy label-free
by construction. Parameters selected on A, evaluated unchanged on B.

Families: global threshold · fixed top-k with floor · threshold + cap ·
leading-score ratio (`p_i ≥ β·p_0`) · successive-score ratio (`p_i ≥ α·p_{i−1}`)
· score gap (`p_{i−1} − p_i ≤ δ`).

**Result:**

| Policy | India B | Δ | US B | Δ |
|---|---|---|---|---|
| **Current global threshold** | **0.9264** | — | **0.9619** | — |
| Best leading-ratio | 0.9259 | −0.0005 | 0.9620 | +0.0000 |
| Best gap | 0.9253 | −0.0010 | 0.9619 | −0.0000 |
| Best successive-ratio | 0.9256 | −0.0008 | 0.9619 | −0.0001 |
| Best threshold + cap | 0.9263 | −0.0001 | 0.9616 | −0.0003 |
| Best top-k | 0.9242 | −0.0021 | 0.9587 | −0.0033 |

**Every deviation negative or zero. Rejected.**

The A column demonstrates exactly why the protocol matters: `lead_ratio` scored
**0.9227 on A** versus 0.9223 for the current threshold, then **−0.0005 on B**.
Judged on A, we would have shipped a regression.

**The more valuable finding — our prioritisation had been wrong.** F₀.₅ by true
cardinality bucket on B:

| bucket | 0 | **1** | 2 | 3 | 4–5 | 6–7 | 8+ |
|---|---|---|---|---|---|---|---|
| India | 0.8945 | **0.8174** | 0.9112 | 0.9331 | 0.9424 | 0.9489 | 0.9412 |
| US | 0.9545 | **0.9002** | 0.9586 | 0.9640 | 0.9691 | 0.9738 | 0.9743 |

**High-cardinality entities are our *best* bucket, not our worst.**
Cardinality-1 is the weak class. The nearest-miss errors *were* concentrated in
high-cardinality entities, but each costs almost nothing in F₀.₅ — missing 1 of
7 matches barely moves a precision-weighted metric. **Counting errors misled us
about where score was lost.**

We also tested two learned decision rules and both **lost**: a per-entity
expected-F₀.₅ optimiser under independence (**−0.0131**) and a LightGBM
cardinality selector trained on score-distribution shape (**−0.0060**). The
oracle gap is irreducible uncertainty, not exploitable structure — choosing *k*
correctly requires the truth, and the score distribution does not contain it.

### 11.6 Experiment 5 — cardinality-1 diagnosis

**Intention.** Experiment 4 relocated the problem to cardinality-1 entities.
Determine *why* they fail.

**Protocol.** Run entirely **locally from cached pair scores** (`scores_*.npz`,
persisted during Exp4) — no Kaggle run required. Cache-to-entity mapping
verified by reproducing blocked recall exactly (0.9250 / 0.9763) with zero
`blocked > true` violations across 30,000 entities.

**Population metrics:**

| | n | TP | FP | precision | recall | F₀.₅ | pred-empty |
|---|---|---|---|---|---|---|---|
| India card=0 | 827 | 0 | 110 | — | — | 0.8936 | 89.4% |
| India card=1 | 786 | 666 | 60 | 0.9174 | 0.8473 | **0.8215** | 14.6% |
| India card≥2 | 13,387 | 44,063 | 1,082 | 0.9760 | 0.8599 | 0.9323 | 1.0% |
| US card=0 | 888 | 0 | 37 | — | — | 0.9628 | 96.3% |
| US card=1 | 835 | 769 | 28 | 0.9649 | 0.9210 | **0.9066** | 7.8% |
| US card≥2 | 13,277 | 46,517 | 482 | 0.9897 | 0.9168 | 0.9666 | 0.3% |

**Three-way failure decomposition for cardinality-1:**

| | India | US |
|---|---|---|
| **true target absent from candidates** | **60.8%** | 27.3% |
| **true target top-ranked but below threshold** | 29.2% | **51.5%** |
| a false candidate outscores the true target | 10.0% | 21.2% |

**The two countries fail for different reasons.** India is retrieval-bound;
US is evidence-bound.

**And case 3 is not what it appears.** In the "false candidate wins" cases the
winning impostor scored a median of only **0.148 (India) / 0.080 (US)** — far
below the 0.65/0.75 thresholds — and **zero** true targets were above threshold
but outranked. These are weak-evidence entities where the ordering among
near-zero scores is arbitrary, not genuine competition failures. Collapsing
cases 1 and 2 as "weak evidence" gives India 39.2%, US 72.7%.

**Stratified by candidate count:**

| India bucket | n | recall | F₀.₅ | % blocked out |
|---|---|---|---|---|
| 61–80 | 532 | 0.8741 | 0.8550 | 7.1% |
| 81–120 | 208 | 0.8125 | 0.7680 | 12.0% |
| >120 | 46 | 0.6957 | 0.6763 | **21.7%** |

Monotone degradation with crowding, driven by blocking.

### 11.7 Experiment 6 — learned normalization — REJECTED

**Intention.** India's cardinality-1 failures are 60.8% retrieval. Test whether
normalisation rules *mined from the training pairs themselves* recover them.

**Leakage control** — stricter than anywhere else in the project, because these
are supervised transformations:
- Mappings mined from **250,000 training entities only**; the 15,000 evaluation
  entities excluded, asserted at runtime.
- **Mutual exclusivity makes the barrier structural**: since no target is
  claimed twice, excluding an eval entity necessarily excludes its true targets
  from mining.
- Eval target strings enter only as TF-IDF inference corpus.

**Method.** For each true training pair, compute the symmetric difference of
token sets. Where each side has exactly **one** unmatched token, the alignment
is unambiguous; count it. Keep mappings with support ≥10 and confidence ≥0.6.

**Two design errors caught before the run**, both of which would have produced
plausible-looking wrong results:

1. **Direction.** Source 1 is the clean deduplicated reference and targets carry
   the corruption, so mappings must run **target → S1**. The reverse is
   one-to-many (one clean token has many corrupted forms) and picking a single
   image of it is wrong.
2. **No transitive closure.** Our first version union-found the mappings, which
   chained `south`, `west`, `east`, `central` → `dillii`. That would have made
   "South Delhi" and "West Delhi" byte-identical strings.

We also added a plausibility guard — a mapping is admissible only if the two
tokens have edit similarity ≥0.55 or the shorter is a ≤3-character abbreviation
sharing the initial. Verified against 17 known cases before launch: it kept
every real variant and rejected `llp→group`, `calcutta→no`, `marine→center`.

**Mined mappings (genuinely good):**

```
India addr:  mh→maharashtra  tg→telangana  gj→gujarat  ka→karnataka
             krnaattk→karnataka  mhaaraassttr→maharashtra  odisha→orissa
India name:  c0nsultants→consultants  lndia→india  mi1ls→mills  pvte→private
US addr:     ohio→oh  texas→tx  maine→me  aveneu→avenue  strete→street
US name:     1lc→llc  5mart→smart  gr0up→group  8akery→bakery
```

Homoglyph corruptions, typos, state abbreviations and transliteration variants,
all learned with no external data.

**Result:**

| | India | US |
|---|---|---|
| Previously-blocked card-1 pairs | 73 | 18 |
| **Recovered** | **9 (12.3%)** | **2 (11.1%)** |
| Overall retrieval recall Δ | +0.0068 | +0.0022 |
| Extra candidates | 176,293 | 99,490 |
| **Candidates per recovered true pair** | **498** | **880** |

**Rejected.** Nine pairs.

A secondary finding refuted our own hypothesis: recovery was **higher on Latin
targets (11.4%) than non-Latin (6.7%)**. The normalisation fixes typos and
abbreviations, not cross-script problems — so there is no case for a
script-gated retrieval path.

### 11.8 Experiment 7 — reverse target→S1 retrieval — REJECTED

**Intention.** Current retrieval is S1 → targets, capped at top-40 per entity. A
target ranked 41st for its true entity is unreachable. Reverse retrieval —
target → S1 — attacks exactly that.

**Methodological note, and the reason this is not the naive implementation.**
Reverse retrieval's difficulty is set by the **S1-side pool size**. Our eval
holds 15,000 entities but production India holds **883,188**. Scoring each
target's top-k against only the eval slice makes the task **59× (India) / 88×
(US) easier than production**. We therefore retrieved against the **full** S1
population in both halves of the experiment:

- **Part A (recovery, exact):** queries are the 3,903 India / 1,222 US
  blocked-out true targets; does the correct entity appear in that target's
  top-k drawn from all 883k?
- **Part B (cost, sampled):** 200,000 random targets against full S1, counting
  new pairs, extrapolated.

**Result — Part A (measured):**

| view / k | India recovered | rate | US recovered | rate |
|---|---|---|---|---|
| name k=1 | 290 | 0.0743 | 177 | 0.1448 |
| name k=10 | 727 | 0.1863 | 421 | 0.3445 |
| **addr k=1** | **664** | **0.1701** | 213 | 0.1743 |
| addr k=3 | 923 | 0.2365 | 314 | 0.2570 |
| addr k=5 | 1,034 | 0.2649 | 368 | 0.3011 |
| addr k=10 | 1,197 | 0.3067 | 424 | 0.3470 |

**Result — cost (extrapolated, labelled as estimate):**

| view / k | India est. new/S1 | recovered | **candidates per recovered** |
|---|---|---|---|
| **addr k=1** | **1.1** | **664** | **25** |
| addr k=3 | 6.3 | 923 | 102 |
| addr k=5 | 13.4 | 1,034 | 195 |
| addr k=10 | 33.0 | 1,197 | 414 |

**Our prior was wrong: reverse retrieval works.** These pairs are genuinely
crowded-out, not signal-less, and `addr k=1` recovers 17% of India's blocked
pairs for +1.1 candidates per entity. The elbow is sharp at k=1 — 4× more
efficient than k=3, 17× more than k=10.

**Rejected on economics.** Best case is +1.28% retrieval recall which, at the
leaderboard-measured 13.5% conversion, is worth roughly **+0.002** — against a
new retrieval view, model retraining, a full re-inference, and a larger audited
candidate set.

We also flagged that the ×20.7 extrapolation is **optimistic**: reverse
retrieval concentrates on high-density common-name clusters that a uniform
random sample under-represents relative to their candidate-generating impact.

**But it also bounds the opportunity:** even at k=10 across both views, ~65% of
India's blocked pairs remain unreachable.

### 11.9 Corruption census — rejected

**Intention.** The data is synthetically corrupted. If the noise is mechanical
it may be invertible.

**Method.** Sampled 14,754 true matched pairs and catalogued the transformations.

| pattern | frequency | example |
|---|---|---|
| state spelled-out ↔ abbreviated | **27.34%** | `NC` ↔ `North Carolina` |
| NULL/None literal | 7.65% | `Wilbur Ave, NULL, Greenwich` |
| domain form | 5.47% | `Federaldiagnosticconsultants.Com` |
| zero-padded number | 5.35% | `1802` → `001802` |
| homoglyph digit | 2.89% | `Sanderson` → `5anderson`, `Falcone` → `Fa1cone` |
| scrambled ordinal | 1.15% | `80th` → `80nd`, `63rd` → `63nd` |
| F/K/A, D/B/A prefix | 1.13% | `Noviecto F/K/A Zenia G. Robertson` |

**Result.** Inverting all of them raised mean address cosine by **+0.061**
(0.798 → 0.858), with individual pairs jumping 0.458 → 0.870. But across the
`min_sim_addr=0.30` floor: **14 pairs rescued, 2 lost, net +12 of 8,887
(+0.14%)**. Recall@40 against a 400k distractor pool moved **+0.0024**.

**The corruptions reduce margin, not reachability** — 94.88% of pairs were
already above the similarity floor.

Notable: the *most striking* corruption (`1037 63rd Street` → `1037 63nd Saint`)
was the second-rarest at 1.15%, while the mundane state-abbreviation swap was
24× more common. **Eyeballing examples would have prioritised exactly the wrong
rule.**

---

## 12. Results and Error Analysis

### 12.1 Leaderboard progression

| Submission | Configuration | Score |
|---|---|---|
| v1 | 20 features, `max_df=0.01`, no exact-key | 0.890575 |
| hybrid | v2 India + v1 US/France | 0.893467 |
| **final** | **43 features, `max_df=0.50`, exact-key** | **0.937** |

### 12.2 The recall→score conversion rate

Isolated by the hybrid submission, which changed **only India**:

```
India is 809,986 / 1,732,544 = 46.75% of the test set
Overall Δ = 0.893467 − 0.890575 = +0.002892
India's own Δ = 0.002892 / 0.4675 = +0.0062
India recall ceiling Δ = 0.9254 − 0.8796 = +0.0458

Conversion = 0.0062 / 0.0458 = 13.5%
```

**One point of blocking recall buys 0.135 points of score.** This single number
governed every subsequent decision: it meant recall-side interventions were
discounted ~7× relative to precision-side ones, which is why Experiments 1 and 2
(features) were prioritised over Experiments 6 and 7 (retrieval), inverting the
intuitive ordering.

### 12.3 Locked-holdout progression

15,000 held-out entities per country; threshold selected on A, reported on B.

| Configuration | India B | US B |
|---|---|---|
| 20 features (baseline) | 0.8975 | 0.9470 |
| + 13 IDF/exact → 33 | 0.9198 | 0.9567 |
| **+ 10 competition → 43** | **0.9266** | **0.9620** |
| **total** | **+0.0291** | **+0.0150** |

Production training validation (pooled, 30k entities/country): **0.9432**,
versus 0.9218 for the 20-feature configuration — **+0.0214**, consistent with
the entity-weighted locked-fold figure of ≈ +0.022. Two independent splits
agreeing is evidence the improvement is real rather than split-specific.

**We projected +0.022 entity-weighted. The leaderboard returned +0.0435.** We
report that as an observed fact. Our local→leaderboard relationship was not
stable across submissions (v1 showed a −0.019 gap; the final run showed the gap
closing or inverting) and we do not claim to explain it.

### 12.4 Where the remaining error is

Final model, locked holdout:

| | true pairs | blocked out | model-rejected | precision | recall | pairwise F₀.₅ |
|---|---|---|---|---|---|---|
| India | 52,027 | 3,903 (**7.50%**) | 3,395 (**6.53%**) | 0.9606 | 0.8088 | 0.9259 |
| US | 51,573 | 1,222 (**2.37%**) | 3,065 (**5.94%**) | 0.9875 | 0.8777 | 0.9634 |

Reconciliation `TP + FN_blocked + FN_model = total true pairs` holds exactly.

Between the audit and the final model the candidate-side loss fell:

```
India  11.61% → 6.81% → 6.53%      (audit 20f → Exp1 33f → Exp2 43f)
US      9.86% → 7.46% → 5.94%
```

**India's loss profile inverted.** Retrieval (7.50%) is now larger than model
loss (6.53%). US remains model-dominated 2.5:1. The two countries now have
genuinely different bottlenecks.

### 12.5 Blocked pairs: what would recover them

The blocked-out population is by construction the complement of the shipped
union, so the four shipped strategies recover 0% of it by definition. The only
other view we built:

| | India | US |
|---|---|---|
| Blocked-out true pairs | 3,903 (7.50%) | 1,222 (2.37%) |
| Recoverable by numeric blocking | 358 (9.2%) | 118 (9.7%) |
| Cost | 708 cands/pair | 2,397 cands/pair |
| **Recoverable by none of the five views** | **3,545 (90.8%)** | **1,104 (90.3%)** |

**~90% of missing pairs are not recovered by any lexical mechanism we built.**
Reverse retrieval reaches 17–31% of them (§11.8); the remainder would need a
fundamentally different representation.

### 12.6 Final output distribution

| country | rows | empty % | mean matches | p95 | max | S2 matches | S3 matches |
|---|---|---|---|---|---|---|---|
| India | 809,986 | ~6.9% | ~3.26 | 6 | 18 | — | — |
| US | 663,106 | ~6.4% | ~3.11 | 6 | 11 | — | — |
| France | 259,452 | ~6.1% | ~3.12 | 6 | 17 | — | — |

France's empty rate (6.1%) now sits between US and India rather than at v1's
16.11% — the open-set path is healthy, not degrading.

---

## 13. Engineering Defects Found and Fixed

Documented because several would have silently corrupted results rather than
failing loudly, and because the class of defect is more instructive than the
instance.

### 13.1 Correctness defects

| # | Defect | Symptom | Root cause |
|---|---|---|---|
| 1 | **Index-offset mismatch** | A multi-country run reported 0.4690 instead of 0.93 | Candidate columns were offset to be globally unique across country partitions; the ground-truth sets were **not** offset to match. Every intersection for the second country was empty, zeroing that country entirely. Now pinned by a regression test. |
| 2 | **SciPy prunes explicit zeros** | Misaligned score arrays, plausible values | `zeros + X` in sparse addition collapses to X's own pattern. Fixed with a sorted int64-key gather plus a length assertion. |
| 3 | **`n_threads=0` runs serial** | ~2.7× slowdown, no warning | `sparse_dot_topn` evaluates `n_threads or 1`. Both 0 and `None` mean serial. |
| 4 | **`exact_key` skipped oversized groups** | Feature delivered +0.67 instead of +2.3 points | The cap excluded precisely the pathological common-name groups the feature existed to handle. |
| 5 | **Polars kwarg renamed across versions** | Worked locally, broke on the hosted runtime | `missing_utf8_is_empty_string` → `empty_string_is_null`. Fixed by runtime signature detection rather than pinning. |
| 6 | **Transliteration changed return type** | `.pipe(...).collect()` broke | A lazy frame was silently converted to eager. Fixed by restoring the input type. |
| 7 | **Feature-space mismatch mid-flight** | A shard died after 113 minutes: *"number of features in data (32) is not the same as it was in training data (20)"* | Kernels cloned `main` at runtime; a feature push landed while shards were queued. Fixed by pinning `COMMIT` with a `rev-parse` assertion. |

### 13.2 Silent-success defects

| # | Defect | Root cause |
|---|---|---|
| 8 | **Unchecked fan-out** | Nine shard pushes, five succeeded (platform concurrency cap), the loop printed "all shards launched" and exited 0. |
| 9 | **Transient treated as fatal** | Auth expiry, then a DNS blip, each killed an unattended launcher that had already waited hours. Fixed by inverting the default to *retry*, giving up only after 20 consecutive unrecognised failures. |
| 10 | **Corrupt archive** | `zip -r - > file` produced a 187 MB archive with a correct name, size and timestamp — and no usable central directory. Fixed by zipping to a real path plus `unzip -t` and per-file presence checks. |
| 11 | **Verification code failing open** | `unzip -l \| grep -q` under `pipefail`: `grep -q` exits on first match, SIGPIPEs the producer, and the pipeline reports failure — so a *present* file was reported missing, but only for names sorting early in the listing. |
| 12 | **Substring guard false positive** | A provenance check forbidding the token `gt` matched `minlen`**`gt`**`h`. Replaced with an AST identifier walk. |
| 13 | **Credential isolation silently ineffective** | `KAGGLE_CONFIG_DIR` was honoured by neither the listing nor the push; operations ran as the default account while appearing to use another. Detected because `--mine` returned the wrong user's kernels. |

### 13.3 Data-handling defects

| # | Defect | Root cause |
|---|---|---|
| 14 | **Merge glob swept two generations** | `merge_partials` globs `matching_results_*.tsv`; an older run's partials in the same directory would merge silently into a file with the correct row count and wrong contents. Fixed by giving each run its own output directory. |
| 15 | **Sampled pool without positives** | An evaluation pool built without the queries' true targets measures the distractor distribution, not recall. Symptom: US "abstained" on 95.43% against a known 6.54%. |
| 16 | **Destructive normalisation via transitive closure** | Union-find over mined token mappings chained `south`/`west`/`east`/`central` → `dillii`. |

---

## 14. Heuristics Discovered

Transferable lessons, each earned from a measurement that contradicted an
expectation.

**H1 — Data-first beats mechanism-first.** Our largest wins came from measuring
the data (script distribution, cardinality, exclusivity) rather than adding
machinery. The highest-gain feature in the final model came from asking *what
can our feature list not express?*

**H2 — Re-validate sampled hyperparameters at full scale.** `max_df=0.01` was
tuned on a 369k pool and cost 2.6 points at full scale.

**H3 — Read the failures, not just the aggregate.** "85% of misses are ranking
failures on byte-identical keys" produced exact-key blocking. No recall number
alone would have suggested a hash join.

**H4 — Decompose the metric before optimising it.** Knowing that *missing >
adding* at the modal cardinality determined the entire precision/recall posture.

**H5 — A cost cap can defeat the feature's purpose.** `max_group` *skipping*
oversized groups removed exactly the cases exact-key blocking existed to catch.

**H6 — Absence of error is not success.** A fan-out step reporting success it
never verified failed us three times.

**H7 — Recall is a ceiling, precision is a dial.** Blocking loss is permanent;
precision is purchasable at the threshold. Bias adjacent stages oppositely.

**H8 — Slice by natural partition, but verify it first.** Country partitioning
is lossless *here* because we checked that no match crosses countries.

**H9 — Prefer guarantees to rankings.** A hash join beats better ranking for
pairs that are trivially matchable but crowded out.

**H10 — Suspect generated data.** The corruption is mechanical and enumerable —
worth cataloguing even though inverting it did not pay.

**H11 — A live problem statement is a moving target.** An update making
candidate-set size a ranking criterion appeared after our initial reading and
changed a blocking decision.

**H12 — Name which axis each hyperparameter moves.** `top_n` controls candidate
*volume*; `max_df` controls candidate *quality*. Conflating them made a sweep
confounded.

**H13 — A platform quota is a silent partial failure**, not an error.

**H14 — A sampled evaluation pool without guaranteed positives measures
nothing.** Watch for an absolute number wildly off a known production figure —
that is the harness being wrong, not the model.

**H15 — Label inferences as inferences.** "France ≈ 0.78" was arithmetic resting
on an assumption, and it misdirected hours before direct measurement.

**H16 — If the decision depends on context, the features must carry the
context.** When post-processing cannot recover a measured gap, suspect the model
is missing an *input* rather than that the post-processor needs to be cleverer.

**H17 — A glob that matches your old output will silently eat it.**

**H18 — Verify an artefact by opening it**, not by seeing that it exists.

**H19 — Verification code needs verifying.** A false negative burns the trust
that makes the check useful.

**H20 — A model and the code that feeds it are one deployable unit.** Pin the
code to the commit that trained the model; treat `main` as unsafe to consume
while anything is in flight.

**H21 — Count-based error analysis misleads under a non-linear metric.** Most
of our absolute errors were on high-cardinality entities, which are our *best*
scoring bucket. Weight error counts by their metric impact before prioritising.

**H22 — Measure the cost side of every recall gain.** "Candidates per recovered
true pair" turned three plausible retrieval ideas into clear rejections.

---

## 15. Reproducibility

### 15.1 Environment

```bash
cd code/business_entity_resolution
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Nine pinned packages: `scikit-learn`, `sparse_dot_topn`, `scipy`, `numpy`,
`polars`, `rapidfuzz`, `Unidecode`, `lightgbm`, `pytest`. Python 3.11+.

Resources: **4 vCPU / 31 GB**. Training ~2.5 h; full test inference ~5 h
sharded nine ways.

⚠️ An earlier `requirements.txt` was a whole-machine `pip freeze` — 250+
packages including `torch`, `sagemaker` and `mlflow` — that both failed to
resolve **and omitted four packages the pipeline cannot run without**
(`lightgbm`, `sparse_dot_topn`, `rapidfuzz`, `Unidecode`). The current file was
rebuilt from the actual import graph and verified by installing into a clean
virtualenv and importing all ten modules.

### 15.2 Data layout

```
dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
dataset/test/{test_source1,test_source2,test_source3}.tsv
```

Overridable via `LASSI_DATA_ROOT`, `LASSI_ARTIFACTS`, `LASSI_OUTPUT`.

### 15.3 End-to-end

```bash
cd src

# 1. Train. 30,000 entities per country is the figure that produced the
#    submitted model, not a placeholder. Writes artifacts/model.txt and
#    artifacts/threshold.json.
python train_model.py --sample-entities 30000

# 2. Predict. The threshold is read from threshold.json automatically, so
#    the two stages cannot drift apart.
python pipeline.py predict --model ../artifacts/model.txt --output-dir ../../output

# 3. Validate.
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

### 15.4 Sharded execution

Test inference is ~5 h of CPU. Any subset works; `merge` reassembles whatever
partials it finds and fails loudly if any test entity is missing.

```bash
python pipeline.py predict --country France
python pipeline.py predict --country India --shard 0 --shards 5
python pipeline.py merge --output-dir ../../output
```

Sharding is by Source-1 entity range; **each shard blocks its slice against the
full country target pool**, so shard count does not affect results — only
wall-clock time.

⚠️ Run `merge` against a directory holding **one run's** partials only (§13.3).

### 15.5 The execution harness

`kaggle/` contains the ten notebooks that produced the submitted outputs. They
are **thin runners** — no pipeline logic — that clone this repository at a
pinned commit and call `src/`. Reproducibility therefore does not depend on any
notebook state.

| Stage | Kernels | Runtime |
|---|---|---|
| Training | 1 | 2 h 26 m |
| India | 5 shards | ~6 h 30 m |
| US | 3 shards | ~5 h |
| France | 1 | ~2 h 15 m |

### 15.6 Test suite

**28 unit tests**, including:
- The problem statement's worked example, asserted to 0.714 and to exact value
- All four cases involving a 0/0 division
- The precision asymmetry at seven cardinalities
- `test_offset_predictions_against_unoffset_truth_scores_zero` — pinning defect #1
- Macro-averaging behaviour, strict/non-strict modes, the all-empty floor

Plus a **synthetic end-to-end harness** (`tests/make_synthetic.py`) generating a
small dataset with every noise pattern the PS names — abbreviation swaps, suffix
inconsistency, `&`/`and`, transpositions, typos, dropped components, landmark
references, non-Latin script — **and a third country present only in test**. It
exercises train → 3-country predict → merge → official validator in seconds.

This harness caught two defects that would otherwise have surfaced only after
multi-hour hosted runs, including an argparse flag that was never registered and
would have killed all nine shards instantly.

---

## 16. Output Validation

| Check | Result |
|---|---|
| `matching_results.tsv` rows | **1,732,544** (exact) |
| `candidate_pairs.tsv` rows | **1,732,544** (exact) |
| Every test Source-1 entity present exactly once | ✅ |
| Duplicate `source1_entity_id` rows | 0 |
| Duplicate IDs within any list | 0 |
| Source-1 self-matches | 0 |
| IDs absent from the test set | 0 |
| Entities with zero candidates | 0 |
| Matches ⊆ candidates | ✅ (enforced structurally — `select()` only ever returns candidate-set members) |
| France present and non-degenerate | ✅ 259,452 rows |
| Official `validate_submission.py --check-ids` | **PASS** |

---

## 17. What We Would Do Next

The measured bottleneck is **retrieval representation**, not the resolver, and
the two countries need different work.

**India — retrieval-bound.** Ceiling 0.9254; 60.8% of cardinality-1 failures are
pairs never retrieved; **90.8% of blocked pairs are unreachable by all five
lexical views**. Reverse retrieval reaches 17–31% of them and is the most
promising untested direction, but needs its candidate-cost problem solved —
likely a *stratified* application (crowded entities only, address view only,
k=1) rather than a global sixth view.

**US — evidence-bound.** Only 2.37% blocked out, but 51.5% of cardinality-1
failures are pairs that are correctly ranked first and still score below
threshold. That is a representation problem in the pairwise scorer, not a
retrieval problem.

**What we would not do first.** Embeddings, neural rerankers, graph features and
stacked models all remain untested — deliberately. Every cheap structural
feature we added (IDF overlap, competition rank) outperformed the expensive
mechanisms we did test, and the residual analysis (§11.4) shows the remaining
false negatives are currently indistinguishable from the false positives we
already accept. Until a representation separates those two populations, a larger
model is unlikely to help.

**Two cheap items we would do immediately:**
1. **Persist pair scores in production** (`--save-scores`, already implemented).
   Every decision-layer question in this project cost a multi-hour re-inference
   because probabilities were computed and discarded.
2. **Resolve exclusivity globally at merge time** rather than per-shard,
   eliminating the 0.66% of contested IDs (§9.3) — which requires (1).
