# ML Challenge 2026 — Business Entity Resolution
## Team Lassi — Methodology Document

**Final leaderboard score: 0.937** (macro F₀.₅)
**Progression: 0.890575 → 0.893467 → 0.937**

---

## 1. Executive Summary

We built a three-stage entity resolution pipeline — **recall-greedy candidate
generation → precision-calibrated pairwise scoring → per-entity set decision** —
and improved it from 0.8906 to **0.937** through a sequence of seven controlled
experiments, three of which we rejected on their own measurements.

The final system is a **43-feature LightGBM** pairwise classifier over a
candidate set built from four complementary blocking views, with a single
global decision threshold and mutual-exclusivity resolution.

### What actually moved the score

| Change | Locked-holdout Δ (India / US) | Kept? |
|---|---|---|
| Union of name **and** address blocking | +14% recall vs name-only | ✅ |
| Transliteration (Indic → Latin) | 34.9% of cross-script pairs made reachable | ✅ |
| `max_df` 0.01 → 0.50 | +2.9 / +2.6 pts recall ceiling | ✅ |
| Exact-key blocking (hash join) | +2.3 / +0.7 pts recall ceiling | ✅ |
| **IDF / rare-token features** | **+0.0223 / +0.0097** | ✅ |
| **Competition (rank/share) features** | **+0.0068 / +0.0053** | ✅ |
| Cardinality-aware decision policies | −0.0005 to −0.0033 | ❌ rejected |
| Learned normalization (mined mappings) | 9 blocked pairs recovered | ❌ rejected |
| Reverse target→S1 retrieval | +1.3% recall @ 25–414 cands/pair | ❌ rejected |
| Numeric-token blocking | +0.7% recall @ 708 cands/pair | ❌ rejected |

**The single most important methodological decision was measuring every idea
against a locked holdout before adopting it.** Four of our ten candidate
improvements were actively harmful or uneconomic, and we would have shipped at
least two of them on intuition alone.

---

## 2. Problem Analysis

### 2.1 The metric determines the architecture

The evaluation is **macro-averaged F₀.₅ per Source-1 entity**, and this single
fact drove nearly every design decision:

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

For a 4-match entity:

| Prediction | Precision | Recall | F₀.₅ |
|---|---|---|---|
| all 4 correct | 1.00 | 1.00 | **1.0000** |
| 3 of 4, no false positive | 1.00 | 0.75 | **0.9375** |
| all 4 **plus one false positive** | 0.80 | 1.00 | **0.8333** |

**Missing a true match scores better than adding a wrong one.** This is not a
tiebreak — it is the dominant gradient of the problem, and it means the
decision stage must be precision-biased even though the candidate stage must be
recall-greedy. Those opposite biases are deliberate and are, we believe, where
most approaches lose score: blocking tightly "to keep precision high" throws
away recall that can never be recovered downstream, and gains nothing, because
precision is bought at the threshold.

We implemented the metric first, unit-tested it against the problem
statement's worked example (0.714) before writing any model code, and pinned
it with 28 tests. Everything downstream is calibrated against that module, so
it had to be right before anything else existed.

### 2.2 The singleton floor

Predicting empty for every entity scores exactly the singleton rate. We
measured this at **5.58%** on training (123,247 of 2,206,821). This is the
trivial floor any model must beat, and it reframes false positives on
singletons as clawing back points already held.

### 2.3 Data facts we measured before modelling

| Statistic | Value |
|---|---|
| train S1 / S2 / S3 | 2,206,821 / 5,034,616 / 5,285,603 |
| test S1 (US / India / France) | 1,732,544 (663,106 / 809,986 / 259,452) |
| Singleton rate | **5.58%** |
| Total matched IDs | 7,638,365 (S2 48.4%, S3 51.6%) |
| Mean matches per S1 | 3.461 (3.666 over non-singletons) |
| **Maximum cardinality** | **11** |
| **Mutual exclusivity** | **PERFECT — 0 of 7,638,365 targets claimed by >1 S1** |
| Target coverage | 74.0% of S2+S3 matched; **26.0% are pure distractors** |
| Address missing | S2 3.356%, S3 3.328% (S1 complete) |
| Exact duplicates (name\|address) | S1 0.00%, S2 0.51%, S3 0.36% |
| Cross-country matches | **none** — country is a lossless hard partition |

Cardinality distribution (share of all S1 entities):

| matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| share | 5.58% | 5.40% | 17.00% | **24.05%** | 21.94% | 14.59% | 7.47% | 2.90% | 1.06% |

---

## 3. Candidate Generation (Blocking)

This stage sets the hard upper bound on recall, and we treated it accordingly:
every parameter below was chosen by measurement against the **full** target
pool, never a sample.

### 3.1 Normalisation pipeline

`src/normalize.py` → `add_normalized_columns()`:

1. **Transliteration** (`unidecode`): Devanagari, Tamil, Telugu, Gujarati and
   Odia → Latin. Source 1 is 100% ASCII while Sources 2/3 are 11–15%
   non-Latin; across that boundary character n-grams score **exactly 0.0**.
2. **Accent folding** — a vectorised Latin-diacritic map (not `unicodedata`
   NFKD, which would force a per-row Python call).
3. **Punctuation normalisation** — `&` → `and`, apostrophes deleted, other
   punctuation → space.
4. **Address abbreviation expansion** — `rd→road`, `ave→avenue`, `ct→court`, …
5. **Legal-suffix removal + token sorting** for the blocking key.

**Token sorting is deliberate.** The problem statement promises word-order
transpositions and address component reordering; sorting makes the key
invariant to both. `Sun Constructions Pvt. Ltd.` and
`SUN CONSTRUCTIONS PRIVATE LIMITED` produce the identical key
`constructions sun`.

**Transliteration, measured on 4,000 cross-script true pairs:**

| key | mean sim | median | >0.15 |
|---|---|---|---|
| raw | 0.286 | 0.000 | 43.1% |
| transliterated | **0.479** | **0.267** | **73.4%** |

**34.9% of cross-script pairs move from effectively zero similarity to
usable.** Without this step those pairs are invisible to blocking regardless
of how the ranking is tuned.

We also derived transliterated legal suffixes empirically from the data
(`limittedd`, `praaivett`, `praaibhett`, `piraiveett`, …) rather than guessing
them — these are corrupted transliterations of "Limited"/"Private" that the
Latin suffix list does not recognise.

### 3.2 Blocking strategies (shipped)

| # | Strategy | Key | Parameters |
|---|---|---|---|
| 1 | Name cosine top-k | char 3-gram TF-IDF, `char_wb` | `top_n=40`, `min_sim=0.25` |
| 2 | Address cosine top-k | char 3-gram TF-IDF, `char_wb` | `top_n=40`, `min_sim=0.30` |
| 3 | Exact name key | hash join on `name_block` | `max_group=100`, truncate |
| 4 | Exact address key | hash join on `addr_block` | `max_group=100`, truncate |

Vectoriser: `min_df=2`, `max_df=0.50`, **fitted on the target corpus** (larger,
more stable IDF; both sides must share one vocabulary for the dot product to be
a cosine at all). Top-k via `sparse_dot_topn.sp_matmul_topn`, chunked at 20,000
rows.

**The candidate set is the UNION, not the intersection.** This is the single
most important choice in the pipeline. Measured over all 7,638,365 training
pairs:

| Source-2/3 name | address | share of true pairs |
|---|---|---|
| Latin | Latin | 79.30% |
| Latin | non-Latin | 6.81% |
| **non-Latin** | **Latin** | **11.63%** |
| non-Latin | non-Latin | 2.26% |

For that 11.63%, name similarity is structurally zero and **only the address
proposes the pair**. Name-only blocking discards them permanently.

### 3.3 Why exact-key blocking exists

Top-k ranking has a structural failure mode that no amount of tuning fixes.
Our miss analysis found **85.4% of US misses and 50.7% of India's** were
"Latin script, good token overlap → *ranking* failure", including pairs whose
normalised keys are **byte-identical**:

```
'Office of Housing'           vs 'Office Of Housing'        jaccard 1.00
'Great Media Private Limited' vs 'Great Media Private Ltd'  jaccard 1.00
```

They are missed because common names have hundreds of key-mates and top-40
cannot hold them all. A hash join has no ranking cutoff: if the keys match, the
pair is proposed. **That converts a ranking into a guarantee**, which is
precisely what the failure mode calls for.

⚠️ **A subtle bug worth documenting.** Our first implementation *skipped* key
groups larger than `max_group`. That removed exactly the common-name cases the
blocker exists to catch, and the measured gain was only +0.67 points. Changing
"skip" to "truncate to a bounded sample" raised it to +2.3 points on India. *A
cap intended to bound cost can silently defeat the feature's entire purpose.*

### 3.4 Parameter sweep (full target pool, 30,000 entities/country)

| config | US ceiling | India ceiling | time |
|---|---|---|---|
| `top_n=40`, `max_df=0.01` | 0.9464 | 0.8863 | 316 s |
| + exact-key | 0.9531 | 0.9093 | 376 s |
| `top_n=100` + exact + `max_df=0.50` | 0.9825 | 0.9486 | 1412 s |
| **shipped: `top_n=40` + exact + `max_df=0.50`** | **0.9757** | **0.9254** | ~600 s |

**`top_n` was deliberately kept at 40.** Raising it to 100 buys ~0.7 further
points of ceiling but roughly doubles both compute and the candidate set. The
two knobs act on different axes — `top_n` controls candidate **volume**,
`max_df` controls candidate **quality** — and since the challenge ranks smaller
candidate sets higher, spending on quality rather than volume is the correct
trade.

⚠️ **`max_df` is a parameter we initially got wrong.** We first set `0.01`,
justified as "a 21× speedup for 1.7 points of recall" — measured on a 369k
sampled pool. Re-measured at full scale the trade was **2.6 points**, not 1.7,
and at `0.01` roughly 70% of each name's n-grams were being discarded.
*Any blocking parameter tuned on a sample must be re-validated at full scale;
a smaller pool has ~10× fewer distractors competing for the same top-k slots.*

### 3.5 Per-strategy attribution (measured, 15,000 held-out entities/country)

| strategy | India recall | US recall | unique contribution |
|---|---|---|---|
| name TF-IDF | 0.5676 | 0.7002 | 1,497 / 1,161 pairs |
| **address TF-IDF** | **0.8097** | **0.8966** | **13,907 / 9,167 pairs** |
| exact name | 0.4445 | 0.5113 | 197 / 206 pairs |
| exact address | 0.1024 | 0.1922 | **0 / 0 pairs** |
| numeric (not shipped) | 0.2689 | 0.1946 | 358 / 118 pairs |
| **union (shipped)** | **0.9250** | **0.9763** | — |

Cumulative: name 0.5676 → +addr **0.9212** → +exact_name 0.9250 → +exact_addr
**0.9250** (no change).

**Address blocking carries the system.** Exact-address contributes zero unique
pairs at `max_df=0.50` — it was worth +2.3 points at `max_df=0.01`, and
relaxing the pruning made it redundant. We retained it because removing it
would change a validated configuration for no measured benefit, but it is
honestly reported here as inert.

### 3.6 Final candidate set

| country | mean/S1 | p50 | p95 | p99 | max | zero-candidate |
|---|---|---|---|---|---|---|
| India | 87.8 | 78 | 123 | 179 | 180 | **0** |
| US | 87.3 | 76 | 178 | 180 | 180 | **0** |
| France | 90.4 | 78 | 179 | 180 | 226 | **0** |

Reduction ratio **0.99998** against the full pair space.

---

## 4. Matching Model and Feature Engineering

### 4.1 Model

**LightGBM binary GBDT.** Chosen because the inputs are ~43 dense, bounded,
tabular similarity scores — precisely the regime where gradient boosting is
strongest — it trains on CPU in minutes, and it uses **no pretrained weights**,
so the MIT/Apache and ≤8B-parameter constraints are satisfied by construction
rather than by argument.

```
objective          binary
learning_rate      0.08        num_leaves        63
min_data_in_leaf   200         feature_fraction  0.9
bagging_fraction   0.8 (freq 1)  lambda_l2       1.0
num_boost_round    600         early_stopping    50
seed 42, deterministic=True
```

Training: **30,000 sampled S1 entities per country**, each blocked against the
**full** country target pool (India 4.13M, US 6.19M). Positive rate 3.69% /
3.70%. Validation `average_precision` **0.9821**.

**Sampling is by entity, never by pair.** The metric is macro-averaged per
entity and the decision stage reasons over an entity's whole candidate list, so
training must see realistic per-entity candidate distributions. Splits are
grouped by S1 entity — a pair from entity X in train and another from the same
entity X in validation would leak, since they share a Source-1 record and often
near-duplicate targets.

**One model for both countries and both target sources.** Source enters only as
the binary feature `tgt_is_s3`. Country is never encoded — see §4.4.

### 4.2 The 43 features

**Group A — 20 pairwise-absolute** (`src/features.py:FEATURE_NAMES`)

| # | Feature | Definition |
|---|---|---|
| 0–4 | `name_token_set`, `name_token_sort`, `name_ratio`, `name_jaro`, `name_cos` | rapidfuzz token-set / token-sort / Indel ratio, Jaro-Winkler, char-3gram TF-IDF cosine |
| 5–9 | `addr_*` | the same five on the normalised address |
| 10–12 | `nums_token_set`, `nums_exact`, `nums_both_present` | over address digit runs |
| 13–16 | `name_len_ratio`, `addr_len_ratio`, `name_tok_diff`, `addr_tok_diff` | structural |
| 17–19 | `tgt_is_s3`, `tgt_name_non_ascii`, `tgt_addr_non_ascii` | regime flags |

The regime flags matter more than they look: for a cross-script pair every name
feature is ≈0, and the flag lets the model learn *"ignore name similarity here,
trust the address"* as a regime rather than reading the zero as disagreement.

**Group B — 13 exact-match and IDF features** (validated +0.0094)

```
exact_name_norm, exact_addr_norm, exact_both_norm,
exact_name_block, exact_addr_block,
name_jaccard, addr_jaccard,
name_idf_jaccard, addr_idf_jaccard,
name_idf_coverage, addr_idf_coverage,
name_max_shared_idf, addr_max_shared_idf
```

- `*_jaccard` = |shared tokens| / |union|
- `*_idf_jaccard` = Σ idf(shared) / Σ idf(union) — IDF-weighted Jaccard
- `*_idf_coverage` = Σ idf(shared) / Σ idf(query tokens)
- `*_max_shared_idf` = highest IDF among shared tokens

Word-level IDF (`log(N/(1+df)) + 1`) is fitted on the **target corpus** — the
same corpus and the same moment blocking fits its character vectoriser, so
train and serve agree by construction.

**Group C — 10 competition features** (validated +0.0053 to +0.0068)

```
rank_name, rank_addr, rank_comb        0 = this entity's best candidate
marg_name, marg_addr, marg_comb        gap to this entity's best (≤ 0)
rel_name, rel_addr                     ratio to this entity's best (≤ 1)
n_cands                                how crowded the field is
share_comb                             this pair's share of the entity's
                                       total similarity mass
```

### 4.3 Why the competition features matter — the key insight

Every one of the original 20 features describes a pair **in isolation**. None
could express whether a candidate was *the best of forty* or *the thirty-seventh
of forty* — yet that is precisely what determines whether it is a match. A 0.7
name cosine means something entirely different as the strongest option in a weak
field than as an also-ran behind several near-identical rivals.

Adding relative features cost **nothing** computationally: everything needed
was already in the candidate arrays that blocking had produced.

Measured feature importance (gain) in the final model:

| India | gain | US | gain |
|---|---|---|---|
| **`addr_idf_jaccard`** | **975,400** | `nums_token_set` | 849,586 |
| `nums_token_set` | 185,630 | **`share_comb`** | **635,435** |
| **`share_comb`** | **159,612** | `name_ratio` | 163,700 |
| `addr_idf_coverage` | 102,232 | `name_jaro` | 138,976 |
| `name_token_set` | 78,737 | `addr_idf_coverage` | 116,463 |

**On India, an IDF feature ranks first overall — above every string-similarity
feature we had. On US, `share_comb` ranks second.** Both of the features we
added are load-bearing, and neither existed in the baseline.

⚠️ **Two features were built, measured, and then deliberately removed.**
`tgt_degree` (how many entities claim this target) and `tgt_margin` are
**target-global**: their denominator is the partition being processed — 30k
entities in training versus 160k–660k in an inference shard. That is a
train/serve scale skew, not leakage, but it makes the feature mean something
different at serve time. We ablated them and shipped the 10 **S1-local**
features, which are scale-invariant ratios. *A feature whose semantics depend
on batch size is a liability even when it improves validation.*

### 4.4 France — open-set handling

France appears only in test (259,452 entities) and never in training. We handle
it by **never encoding country as a value anywhere in the pipeline**:

- No country feature, no one-hot, no country-specific branch.
- Country is used **only** as a partition key, which is lossless because we
  verified that no true match crosses countries.
- Every component is structurally language-agnostic: character n-grams degrade
  gracefully across scripts, numeric tokens survive transliteration, and token
  sorting is order-invariant.
- Phonetic keys (Soundex/Metaphone) were **deliberately rejected** — they are
  English-phonology specific and would mangle Indian and French names.

We validated the open-set path with a synthetic country ("Zephyria") present
only in the synthetic test split, exercising the unseen-country path end to end
before real data was used.

**Result: France behaves normally.** Under v1 it abstained on 16.11% of
entities; under the final model it emits at essentially the same rate as the
training countries.

### 4.5 Decision stage

```python
keep = probability >= threshold          # 0.675, swept on validation
resolve_exclusivity(...)                 # one target → at most one entity
```

Threshold selected by direct macro-F₀.₅ optimisation on held-out **entities**,
not by pairwise F1 or accuracy — a threshold chosen by pairwise F1 would sit
near 0.5; optimising the actual metric places it at 0.675.

**Exclusivity** exploits the measured fact that ground truth assigns each S2/S3
record to at most one S1 entity (0 violations in 7.6M pairs). Measured effect:
only 16/43,806 (India) and 3/45,838 (US) assignments changed. We report this
honestly as **near-null but correct and free**.

---

## 5. Results

### 5.1 Leaderboard

| Submission | Score |
|---|---|
| v1 (20 features, `max_df=0.01`) | 0.890575 |
| v2 India hybrid | 0.893467 |
| **Final (43 features)** | **0.937** |

### 5.2 Locked-holdout progression

15,000 held-out entities per country, threshold chosen on split A and evaluated
unchanged on split B.

| Configuration | India B | US B |
|---|---|---|
| 20 features (baseline) | 0.8975 | 0.9470 |
| + 13 IDF/exact → 33 | **0.9198** | **0.9567** |
| + 10 competition → 43 | **0.9266** | **0.9620** |
| **total gain** | **+0.0291** | **+0.0150** |

We projected +0.022 entity-weighted from these numbers. The leaderboard
returned **+0.0435** — roughly double. We report that as an observed fact, not
as a validated relationship; the local→leaderboard mapping was not stable
across our submissions and we never explained it.

### 5.3 Where the remaining error is

Measured on the locked holdout with the final model:

| | true pairs | blocked out | model-rejected | precision | recall |
|---|---|---|---|---|---|
| India | 52,027 | 3,903 (**7.50%**) | 3,395 (**6.53%**) | 0.9606 | 0.8088 |
| US | 51,573 | 1,222 (**2.37%**) | 3,065 (**5.94%**) | 0.9875 | 0.8777 |

Reconciliation check: `TP + FN_blocked + FN_model = total true pairs`, exact in
both countries.

**Per true-cardinality bucket (F₀.₅ on holdout B):**

| bucket | 0 | 1 | 2 | 3 | 4–5 | 6–7 | 8+ |
|---|---|---|---|---|---|---|---|
| India | 0.8936 | **0.8215** | 0.9112 | 0.9331 | 0.9424 | 0.9489 | 0.9412 |
| US | 0.9628 | **0.9066** | 0.9586 | 0.9640 | 0.9691 | 0.9738 | 0.9743 |

**Cardinality-1 entities are the weakest class**, not high-cardinality ones.
This inverted our expectation: the *absolute count* of errors was highest among
high-cardinality entities, but each such error costs almost nothing in F₀.₅
because 6-of-7 recall is nearly as good as 7-of-7. Counting errors misled us
about where score was actually lost.

**Cardinality-1 failure decomposition:**

| | India | US |
|---|---|---|
| true target absent from candidates | **60.8%** | 27.3% |
| true target top-ranked but below threshold | 29.2% | **51.5%** |
| a false candidate outscores the true target | 10.0% | 21.2% |

The two countries fail for different reasons: **India is retrieval-bound, US is
evidence-bound.** Notably, in the "false candidate wins" cases the winning
impostor scored a median of only 0.148 (India) / 0.080 (US) — far below
threshold — so these are weak-evidence entities where the ordering among
near-zero scores is arbitrary, not genuine competition failures.

---

## 6. Experiments We Rejected

We consider these as important as the ones we kept. Each was fully implemented
and measured before being discarded.

### 6.1 Cardinality-aware decision policies — rejected

We tested **236 parameterisations across 6 policy families** (fixed top-k,
threshold+cap, leading-score ratio, successive-score ratio, score-gap, and the
current global threshold), selecting on split A and evaluating unchanged on B.

| policy | India B Δ | US B Δ |
|---|---|---|
| current global threshold | — | — |
| best leading-ratio | −0.0005 | +0.0000 |
| best threshold+cap | −0.0001 | −0.0003 |
| best top-k | −0.0021 | −0.0033 |

**Every deviation was negative or zero.** The model's pair ordering is not a set
ordering, and no reweighting of *where to cut* that ordering recovers anything.
The global threshold is genuinely optimal.

We also tested a **learned cardinality selector** (LightGBM predicting the
optimal prefix length from the score distribution's shape) and an **expected-F₀.₅
optimiser** under independence. Both scored *worse* than a flat threshold
(−0.0060 and −0.0131). The oracle gap is irreducible uncertainty, not
exploitable structure — choosing *k* correctly requires the truth, and the score
distribution does not contain it.

### 6.2 Learned normalization — rejected

We mined **target→S1 token rewrite rules** from training pairs only (250,000
entities, eval excluded; mutual exclusivity makes the barrier structural since
excluding an entity also excludes its targets). The mined mappings were
genuinely good:

```
India addr:  mh→maharashtra  tg→telangana  gj→gujarat  krnaattk→karnataka
India name:  c0nsultants→consultants  lndia→india  pvte→private
US addr:     ohio→oh  texas→tx  aveneu→avenue  strete→street
US name:     1lc→llc  5mart→smart  gr0up→group  8akery→bakery
```

**But they recovered only 9 (India) and 2 (US) previously-blocked
cardinality-1 pairs**, at 498 / 880 additional candidates per recovered true
pair. Rejected on cost.

Two implementation notes that may be useful to others:
- **Direction matters.** S1 is the clean deduplicated reference and the targets
  carry corruption, so mappings must run target→S1. The reverse is one-to-many.
- **Do not take transitive closure.** Our first version union-found the
  mappings, which chained `south`, `west`, `east`, `central` → `dillii`. That
  would have made "South Delhi" and "West Delhi" byte-identical.

### 6.3 Reverse target→S1 retrieval — rejected on economics

Measured honestly against the **full** S1 population (883,188 India /
1,323,633 US), not the 15k evaluation slice — scoring against the small slice
would have made the task 59×/88× easier than production.

| view / k | India recovered | cands per recovered |
|---|---|---|
| addr k=1 | 664 (17.0%) | **25** |
| addr k=3 | 923 (23.7%) | 102 |
| addr k=5 | 1,034 (26.5%) | 195 |
| addr k=10 | 1,197 (30.7%) | 414 |

It **works** — it recovers 17–31% of blocked pairs, and our prior that it would
fail was wrong. But the best case is +1.28% retrieval recall for a 1.3%
candidate-set increase, which at the measured recall→score conversion is worth
roughly +0.002. Rejected against the cost of integration, retraining, and a
larger audited candidate set.

### 6.4 Numeric-token blocking — rejected

Implemented but never enabled. Marginal contribution: **+358 true pairs for
253,409 extra candidates (708 per pair)** on India; 2,397 per pair on US.

### 6.5 Corruption inversion — rejected

We catalogued the generator's noise on 14,754 true pairs: state
spelled-out↔abbreviated **27.34%**, NULL/None literal 7.65%, domain form 5.47%,
zero-padding 5.35%, homoglyph digits (`5anderson`, `Fa1cone`) 2.89%, scrambled
ordinals (`80th`→`80nd`) 1.15%.

Inverting all of them raised mean address cosine by **+0.061** but recall@40 by
only **+0.0024** — because 94.88% of pairs were already above the similarity
floor. **The corruptions reduce margin, not reachability.**

---

## 7. Heuristics Discovered

These are the transferable lessons, each earned from a measurement that
contradicted an expectation.

**H1 — Data-first beats mechanism-first.** Our largest wins came from measuring
the data (script distribution, cardinality, exclusivity) rather than from
adding machinery. The single biggest feature in the final model
(`addr_idf_jaccard`) came from asking *what can our feature list not express?*

**H2 — Re-validate sampled hyperparameters at full scale.** `max_df=0.01` was
tuned on a 369k pool and cost 2.6 points at full scale. A smaller pool has
~10× fewer distractors competing for the same top-k slots.

**H3 — Read the failures, not just the aggregate.** "85% of misses are ranking
failures on byte-identical keys" is what produced exact-key blocking. No amount
of staring at a recall number would have suggested a hash join.

**H4 — Decompose the metric before optimising it.** Knowing that
`missing a match > adding a false one` at the modal cardinality determined the
entire precision/recall posture.

**H5 — A cost cap can defeat the feature's purpose.** `max_group` *skipping*
oversized key groups removed exactly the common-name cases exact-key blocking
existed to catch.

**H6 — Absence of error is not success.** A fan-out step that reports success it
never verified failed us three times: unchecked shard pushes, auth expiry
treated as fatal, and a corrupt archive with a plausible size.

**H7 — Recall is a ceiling, precision is a dial.** Blocking loss is permanent;
precision is purchasable at the threshold. Bias the two stages oppositely.

**H8 — Slice by natural partition.** Country partitioning is lossless here
(verified: no cross-country matches) and made the whole problem tractable.

**H9 — Prefer guarantees to rankings.** Exact-key blocking beats better ranking
for the class of pairs that are trivially matchable but crowded out.

**H10 — Suspect generated data.** The corruption is mechanical and enumerable.
This was worth cataloguing even though inverting it did not pay.

**H11 — A live problem statement is a moving target.** An update banner making
candidate-set size a ranking criterion appeared after our nine logged chunks,
and changed a blocking decision.

**H12 — Name which axis each hyperparameter moves.** `top_n` controls candidate
*volume*; `max_df` controls candidate *quality*. Conflating them made a
confounded sweep look like one knob.

**H13 — A platform quota is a silent partial failure.** Kaggle's 5-session cap
rejected 4 of 9 shard pushes while the loop reported success.

**H14 — A sampled evaluation pool without guaranteed positives measures
nothing.** A pool built without the queries' true targets reports the distractor
distribution, not recall.

**H15 — Label inferences as inferences.** "France ≈ 0.78" was arithmetic resting
on an assumption, and it drove hours of misprioritised work before direct
measurement showed France's blocking was fine.

**H16 — If the decision depends on context, the features must carry the
context.** When post-processing cannot recover a measured gap, suspect the model
is missing an *input* rather than that the post-processor needs to be cleverer.

**H17 — A glob that matches your old output will silently eat it.** Merging by
filename pattern into a directory holding a previous run produces a file with
the right row count and the wrong contents.

**H18 — Verify an artefact by opening it.** `zip -r - > file` produced a 187 MB
archive with no usable central directory. Right name, right size, unopenable.

**H19 — Verification code needs verifying.** `unzip -l | grep -q` under
`pipefail` reported a present file as missing, because `grep -q` exits early and
SIGPIPEs the producer.

**H20 — A model and the code that feeds it are one unit.** Our kernels cloned
`main` at runtime; pushing a feature change while jobs were queued killed a
shard 113 minutes in with a 32-vs-20 feature mismatch. Pin the code to the
commit that trained the model.

---

## 8. Reproducibility

```bash
cd code/business_entity_resolution
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
# 1. Train (30,000 entities per country is the figure that produced the
#    submitted model, not a placeholder). ~2.5 h on 4 vCPU.
python train_model.py --sample-entities 30000

# 2. Predict. Shardable by country; merge reassembles.
python pipeline.py predict --country India --shard 0 --shards 5
python pipeline.py merge --output-dir ../../output
```

**Verification built into the repo:**
- 28 unit tests, including the problem statement's worked example (0.714) and a
  regression test pinning an index-offset bug that once silently zeroed an
  entire country's score.
- A synthetic dataset generator injecting every named noise pattern plus an
  unseen country, exercising train → 3-country predict → merge → official
  validator in seconds.
- The official `validate_submission.py` passes with `--check-ids`.

**Environment:** 4 vCPU / 31 GB. Training ~2.5 h; full test inference ~5 h
sharded 9 ways.

⚠️ `n_threads` must never be 0 or `None` — `sparse_dot_topn` treats both as
serial (`n_threads or 1`), a silent ~2.7× slowdown.

---

## 9. Output Validation

| Check | Result |
|---|---|
| Rows in `matching_results.tsv` | **1,732,544** (exact) |
| Rows in `candidate_pairs.tsv` | **1,732,544** (exact) |
| Every test S1 present exactly once | ✅ |
| Duplicate rows | 0 |
| Duplicate IDs within a list | 0 |
| S1 self-matches | 0 |
| IDs absent from the test set | 0 |
| Zero-candidate entities | 0 |
| France present and non-degenerate | ✅ |
| Official validator (`--check-ids`) | **PASS** |

**One honest caveat:** mutual-exclusivity resolution runs *within* each shard.
Because India is sharded 5 ways and US 3 ways, a target claimed by entities in
different shards survives. On the v2 run this affected 0.66% of emitted IDs.
It is not a rejection condition, and fixing it correctly requires persisted
pair scores to decide which entity keeps a contested target — a capability we
added (`--save-scores`) after the production run had already started.

---

## 10. What We Would Do Next

The measured bottleneck is **retrieval representation**, not the resolver:

- India's blocking ceiling is 0.9254, and **90.8% of the pairs it misses are
  unreachable by all five lexical views** we tested (name, address, exact-key
  ×2, numeric). Those pairs have no usable character- or token-level signal.
- Reverse retrieval recovers 17–31% of them and is the most promising untested
  direction, but needs the candidate-cost problem solved — likely via a
  targeted, stratified application rather than a global sixth view.
- US is evidence-bound rather than retrieval-bound (51.5% of its cardinality-1
  failures are correctly-ranked pairs scoring below threshold), so US and India
  warrant different investments.

We would not add embeddings, neural rerankers, or graph features before
exhausting that, on the evidence that every cheap structural feature we added
outperformed the expensive mechanisms we tested.
