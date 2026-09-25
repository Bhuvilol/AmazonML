# LEARNINGS — every observation, in detail

Running record of what was measured, what was believed and disproved, and what
it cost. Kept because six confident predictions were overturned by measurement
in a single session, and the pattern matters more than any individual result.

Companions: `CONTEXT.md` (state + facts), `ps.md` (problem statement),
`brainstorm.md` (strategy), `Documentation_template.md` (submission write-up).

---

## ★ HEURISTICS — the transferable lessons

These are the rules I would give another engineer starting this problem. Each
one was paid for with a wrong turn today.

### H1. Data-first beats mechanism-first. Every time.
Scorecard for this project: **9 predictions reasoned from a plausible mechanism
were wrong. 3 findings read directly off the data were right and large.**

| source | outcome |
|---|---|
| "singletons must be a big lever" | wrong (5.58%) |
| "mutual exclusivity must buy precision" | wrong (0.0000) |
| "address is the bridge for cross-script" | wrong for India |
| "numeric tokens must help India" | wrong (+0.004) |
| "France abstention is the threshold" | wrong (16%→14.5%) |
| **counted suffix tokens in the data** | **right — limittedd 40%** |
| **read 100 actual missed pairs** | **right — 85% are ranking failures** |
| **re-measured max_df at real scale** | **right — +2.6 points** |

If a hypothesis can be checked in 15 minutes, check it. Do not build on it.

### H2. A hyperparameter tuned on a sample MUST be re-validated at full scale.
`max_df=0.01` cost 1.7 points on a 369k pool and **2.6 points on the real 6.19M
pool** — and the sample made it look like a free 21x speedup. Sampling changes
the *difficulty* of a task, not just its size. This single error probably
accounted for most of our gap to the leaderboard.

### H3. Read the failures. Do not theorise about them.
Dumping 100 missed pairs side-by-side took ten minutes and overturned the whole
diagnosis. We found pairs with **byte-identical normalised keys** being missed —
something no amount of reasoning about cross-script or DBA names would have
surfaced.

### H4. Decompose the metric before optimising it.
Measuring precision and recall separately proved that **even perfect precision
caps us at 0.952** — which killed model changes, feature work, and threshold
tuning as viable directions in one measurement. Know which half you are losing.

### H5. A cap or filter can defeat the very case a feature exists for.
Exact-key blocking was built to catch common names like "office of housing",
then a `max_group=100` cap **skipped keys with many mates** — i.e. exactly those
names. Gain fell to +0.67. When adding a guard, ask which cases it removes.

### H6. Absence of an error is not evidence of success.
A truncated 632 MB download and an expired auth token both looked like success
to code that only checked for the absence of a failure string. Verify by
**content or size**, never by exit code alone, when a network is involved.

### H7. Recall is a ceiling; precision is a dial.
In a two-stage matcher, blocking sets a hard bound nothing downstream can
exceed. Tune the two stages with **opposite** biases: recall-greedy generation,
precision-biased decision. Conflating them is the classic error.

### H8. When the metric is macro and per-entity, per-entity behaviour matters.
Country-level abstention rates (US 6.54%, India 8.58%, France 16.11% against a
5.58% true rate) exposed a real defect that aggregate scores hid completely.
Always slice by the natural partition.

### H9. Prefer guarantees to rankings where you can afford them.
A hash join on an exact key has no top-k cutoff. Similarity ranking silently
drops true matches when a key is common. Use both.

### H10. Suspect generated data, and look for the generator.
`"-- Holloway Peak Inc Seafood"`, `16rd Saint` for `16th Street`, `CORNELISU`
for `Cornelius`. These are *mechanical* corruptions. If noise is synthetic,
inverting the transforms beats fuzzy matching — and it explains leaderboard
scores (0.9859) that are implausible for genuine real-world ER.

**H11 — a live problem statement is a moving target; re-read it, don't recall
it.** The organisers added an update banner to the top of the portal PS
declaring that candidate-set size is a ranking criterion. Nine chunks of PS text
had been logged verbatim in `ps.md` and treated as complete; the banner was
never in them. An open question was sitting unresolved in `ps.md` ("is
`candidate_pairs.tsv` judged?") that the banner answered directly. Cost: a
blocking config chosen against the wrong objective function, caught only because
the user re-sent the PS unprompted. **Re-read the live rules at every phase
boundary, and treat any question marked "unclear" as a standing action item, not
a footnote.**

**H12 — a swept table is only as useful as its unconfounded cells.** The v2
sweep varied `top_n` and `max_df` together, and I described the result as "the
recall config" as though it were one knob. It is two, and they act on different
axes: `top_n` sets candidate *volume* (how many neighbours per entity),
`max_df` sets candidate *quality* (which n-grams carry weight). Under a rule
that penalises volume, that distinction is the whole decision. I nearly reported
a conflict that did not exist because I recalled the config from a summary
instead of reading `BlockingConfig`. **Name which axis each hyperparameter moves
before trading them off, and read the config, don't remember it.**

**H13 — a platform quota is a silent partial failure, not an error.** Kaggle
caps concurrent batch CPU sessions at 5. `chain_v2.sh` pushed nine kernels in
one burst; the five India shards took the slots and the four US/France pushes
were rejected with "Maximum batch CPU session count of 5 reached". The loop
piped each push through `head -1` and then printed "all shards launched"
unconditionally, so the run exited 0 and looked complete. Caught only by
querying each kernel's status by hand. This is **H6 (absence of error is not
success) recurring in a new place** — the second time in this project that a
fan-out step reported success it had not verified. **When a step fans out to N
things, assert N successes; never let the loop's own exit code stand in for the
work's.** Fixed by grepping each push for "successfully pushed", counting, and
exiting non-zero with the deferred list.

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

## 1b. ★ DEFINITIVE: precision is fine, RECALL is the entire problem

Measured on 15,000 held-out training entities per country, using the deployed
model and threshold:

| | US | India |
|---|---|---|
| macro F₀.₅ | 0.9373 | 0.8829 |
| **micro precision** | **0.9812** (869 FPs) | **0.9515** (2,136 FPs) |
| **micro recall** | **0.8759** (6,423 missed) | **0.8044** (10,180 missed) |
| entities perfectly reconstructed | 64.8% | 49.9% |
| **macro F₀.₅ if precision were PERFECT** | **0.9521** | **0.9150** |

**The last row is the decisive number.** Removing every single false positive
still leaves us at 0.952 / 0.915 — below the top-500 cutoff of 0.95 overall.
**No amount of precision work can reach the target.** Recall is the only axis.

### A second loss, previously unisolated
US blocking recall is 0.9462 but end-to-end recall is 0.8759. So:
- ~5.4% of pairs are lost at **blocking** (never proposed)
- ~7.0% more are lost at the **decision stage** (proposed but scored below 0.575)

The threshold sweep says 0.575 is optimal *for this candidate set*, so the
second loss is not a threshold error — the model cannot rank well what blocking
barely surfaced. Both losses trace back to candidate quality.

### What this kills
- **Changing the model is pointless.** AP 0.9878, precision 0.98. A transformer,
  an ensemble, better features — all operate on pairs that must already exist.
- **Per-entity expected-F selection**: at best worth the precision headroom,
  i.e. ≤0.015 on US. Not the answer.
- **France threshold tuning**: still worth ~0.01, but not a path to 0.95.

### What remains
Raise blocking recall from ~0.93 toward ~0.98. Nothing else can move the score
enough. Ranked hypotheses in section 6.

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
| 8 | France abstention is threshold-driven, worth ~0.01 | **Wrong.** Threshold 0.575→0.20 moves it only 16%→14.5%. It is a BLOCKING failure | Would have wasted a submission |

**The pattern: a plausible mechanism, asserted confidently, contradicted by
measurement.** Every single one was caught only by measuring. The process
works; the instinct to trust the mechanism does not.

---

## 3b. ★ FRANCE THRESHOLD HYPOTHESIS: DEAD (prediction #8 wrong)

I argued the France abstention (16.11% vs US 6.54%) was the threshold being too
conservative on a country with no training data, worth ~0.01. Measured curve:

| threshold | empty% |
|---|---|
| 0.200 | **14.52%** |
| 0.300 | 15.11% |
| 0.400 | 15.61% |
| 0.550 | 16.05% |

Dropping the threshold from 0.575 to **0.20** moves abstention only 16% → 14.5%.
**Abstention is not threshold-driven.** Those ~41,000 entities have candidates
scoring below even 0.20, so blocking surfaced the WRONG candidates — the true
matches are absent from the candidate set entirely.

France blocking produced only 256 rows with zero candidates, yet 41,794 predict
empty. So candidates exist; they are simply not the right ones.

**Same root cause as everything else: blocking recall.**

Side observation: exclusivity drops **12.2%** of claims at threshold 0.2 for
France (vs 0.01% in training). Many contested targets is itself a symptom of
poor candidate quality.

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

**H1 — max_df destroyed recall at scale. CONFIRMED, +2.6 points.**

| max_df | recall | secs |
|---|---|---|
| 0.01 (was deployed) | 0.9464 | 212 |
| 0.10 | 0.9712 | 626 |
| **0.50** | **0.9728** | **597** |

0.5 costs no more than 0.1, so the aggressive setting bought nothing past the
first step. **This was the single largest error of the project.**

**H2 — top_n too low. CONFIRMED.** Note the interaction: at `max_df=0.01`
blocking time is flat in `top_n`, but once pruning is relaxed it is NOT
(597 s @40 vs 1412 s @100). More surviving n-grams means more to rank.

### ★ FULL SWEEP — real target pools, both countries

| config | US | India |
|---|---|---|
| top_n=40, max_df=.01 *(was deployed)* | 0.9464 | 0.8863 |
| + exact-key | 0.9531 (+0.7) | 0.9093 (**+2.3**) |
| top_n=100 + exact | 0.9659 | 0.9261 |
| **top_n=100 + exact + max_df=0.5** | **0.9825** | **0.9486** |

**Exact-key blocking helps India 3x more than US** (+2.3 vs +0.7) — Indian names
carry more systematic variants that normalise to identical keys.

These India figures **exclude transliteration** (the measurement process had
already imported the module before it was added). With 42% of India's misses
being cross-script and transliteration moving 34.9% of those from 0 to usable,
India should land materially higher in the real run.

**Deployed v2 config: top_n=40, max_df=0.5, exact-key on, transliteration on.**
top_n=40 over 100 costs ~0.25 points of recall but halves the compute (9 sharded
kernels instead of 18), which is the difference between fitting the deadline and
not.

**H2b — exact-key blocking. PARTIAL: +0.67 points, then a bug found.**
First measurement gave only 0.9464 → 0.9531 because `max_group=100` *skipped*
keys with many mates — precisely the common names ("office of housing") the
blocker exists to catch. Now truncates instead of skipping. See heuristic H5.

**H3 — the noise is GENERATED, not natural.** Evidence: `"-- Holloway Peak Inc
Seafood"`, systematic legal-suffix swaps, controlled typos, token
transpositions, Devanagari transliterations. If the organisers synthesised the
corruption with a fixed transform set, then **explicitly inverting each
transform** beats general fuzzy matching by a wide margin. A 0.986 top score is
implausibly high for genuinely messy real-world ER — this is the theory that
best explains the top of the leaderboard.

**H4 — transliteration. CONFIRMED and BUILT.**

Measured on 4,000 true pairs with a non-Latin target name:

| key | mean | median | >0.15 |
|---|---|---|---|
| raw | 0.286 | **0.000** | 43.1% |
| transliterated (`unidecode`) | 0.479 | **0.267** | 78.7% |

**34.9% of cross-script pairs move from ~0 to usable.** The data contains
Devanagari, Tamil, Telugu, Gujarati and Odia — more scripts than the two
originally identified.

Romanisation is phonetic, not translation ('राम' → 'raam'), which suffices for
blocking. Crucially, the Latin suffix list does NOT recognise the romanised
forms, so they had to be derived by counting tokens in 60,000 transliterated
names: `limittedd` 40.0%, `praaivett` 22.7%, `praiveett` 7.1%, `li` 5.3%,
`elelpii` (LLP) 2.6%. Adding them lifted a worked example from 0.20 to 0.53.

`unidecode` is a character mapping table — an algorithm, not a lookup of
business identities — so it is consistent with the external-data rule.

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
