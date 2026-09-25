# Problem Statement — Working Doc

Live intake document. Updated as each chunk of the PS arrives.

**Rules I'm holding to during intake:**
- Log each chunk raw, then what I read off it.
- No architecture, no model names, no solutioning until the PS is complete.
- Inference is labelled as inference. Spec is labelled as spec. Never blurred.
- When a later chunk contradicts an earlier inference, the correction is
  recorded, not quietly overwritten.

---

## Status

| | |
|---|---|
| Chunks received | 9 |
| PS complete | **NO** |
| Execution allowed | **NO** — blocked until PS complete |
| Task family | Entity Resolution / record linkage |
| Metric | **F₀.₅ — precision-weighted.** Exact formula pending |

---

## Objective

Given business records from **3 independent sources** with noisy inconsistent
fields and **no shared identifiers**, determine which records refer to the same
real-world business.

Source 1 is the deduplicated reference. For every S1 entity, find all matching
records in S2 and S3. An S1 entity may match **zero, one, or many** records.

**What winning means:** a submission that scores. Solo, 2-4 day clock, AWS with
sponsored credits.

---

## The metric: F₀.₅, macro-averaged — this reframes everything

**Fully confirmed in chunk 7.**

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed **per Source 1 entity, then averaged** across all S1 entities in the
evaluation set. Macro, not micro.

### ⚠️ Correction to my earlier framing

I earlier described the weighting as **"1:4 in precision's favour."** The PS
states **2×**, and the PS's phrasing is the conventional one: in F_β, recall is
given β times as much importance as precision, so β = 0.5 makes precision twice
as important.

Where my 4:1 came from: F_β is a weighted harmonic mean whose weights are
`1/(1+β²) = 0.8` on precision and `β²/(1+β²) = 0.2` on recall — a 4:1 ratio in
that specific sense. Both numbers describe the same formula, but **2× is the
standard statement and the one to use.** The practical direction is unchanged:
be conservative, false positives hurt more.

### Why this inverts the standard ER instinct

Conventional ER wisdom: push recall hard in candidate generation, because a
pair you never generate can never be recovered. That still holds for the
*candidate* stage. But at the *decision* stage F₀.₅ punishes false positives
roughly four times harder than it rewards catching an extra true match.

Consequences:

- **The abstain threshold should be conservative.** When in doubt, emit
  nothing. A wrong id costs more than a missed one.
- **Predicting empty is cheap; predicting wrong is expensive.** This makes the
  zero-match case strategically useful rather than merely a formatting
  requirement.
- **Threshold tuning is not a finishing touch, it is a primary lever.** With a
  precision-weighted metric, the same candidate set and the same scorer can
  swing a large amount of final score purely on where the cut sits.
- **Recall built in blocking is still mandatory** — the ceiling argument is
  unchanged. Generate broadly, then decide conservatively. Those are different
  stages with opposite biases.

### Macro-average + singleton rule — the highest-leverage fact in the PS

Confirmed, verbatim:

> Singletons are included in that average. A Source 1 entity with no true
> matches scores **1.0** when you correctly predict an empty list, and **0.0**
> when you predict any match for it.

Three consequences, all large:

1. **Every S1 entity is worth exactly the same**, whether it has zero matches or
   ten. Macro averaging removes any benefit from doing well on the
   high-cardinality entities. An entity with one true match carries the same
   weight as one with eight.

2. **Singletons are free score, or total loss — nothing in between.** Predict
   empty correctly: 1.0. Predict anything at all: 0.0. No partial credit. If
   singletons are a meaningful share of S1, then **singleton detection alone is
   worth that share of the total score, with zero matching skill involved.**

3. **The per-entity all-or-nothing structure on singletons makes the abstain
   decision the single biggest lever in the build**, ahead of the scorer.

### The trade-off, worked out

Macro scoring makes the precision bias concrete. For an entity with **2 true
matches**:

| Prediction | P | R | F₀.₅ |
|---|---|---|---|
| Both correct | 1.0 | 1.0 | **1.000** |
| Both correct **+ 1 false positive** | 0.667 | 1.0 | **0.714** |
| Only 1 of the 2 correct | 1.0 | 0.5 | **0.833** |

> **Missing a true match (0.833) scores better than catching both but adding one
> false positive (0.714).**

That single comparison is the whole strategy. For an entity with **1 true
match**:

| Prediction | P | R | F₀.₅ |
|---|---|---|---|
| The correct one | 1.0 | 1.0 | **1.000** |
| Correct one + 1 false positive | 0.5 | 1.0 | **0.556** |

One spurious id nearly halves that entity's score. And the PS's own worked
example checks out: P = 2/3, R = 1.0 → 0.8333 / 1.16667 = **0.714**.

**Practical upshot:** a marginal candidate needs to clear a confidence bar well
above 50% to be worth emitting. The exact break-even depends on how many true
matches the entity has, which is itself unknown at prediction time — that
asymmetry is worth modelling explicitly rather than tuning one global threshold
by feel.

---

## Public vs private leaderboard — do not tune against the public score

Chunk 8:

- **Public LB** — a *subset* of the test set, live during the challenge.
- **Private LB** — the *remaining* portion, revealed at the end.
- **Final rankings come from the private leaderboard.**
- Predictions are submitted for the **full** test set in both cases; the split
  is applied at scoring time.

### Why this is dangerous here specifically

This is the standard public/private trap, made sharper by two features of this
particular competition:

1. **The metric is macro-averaged over entities.** On a subset, a macro average
   has meaningfully higher variance than a pooled micro metric would — each
   entity contributes equally, so a smaller entity count means a noisier score.
   The smaller the public subset, the noisier the number.

2. **The abstain threshold is the single biggest lever** (see the metric
   section). Tuning the highest-leverage parameter against the noisiest
   available signal is precisely how a strong local model loses on private.

> **Rule: tune the threshold on local CV. Use the public leaderboard to confirm
> format correctness and catch gross errors — not as the optimisation target.**

### Unknown, and it matters

- **The public/private ratio is not stated.**
- **Whether the split is stratified by country is not stated.** If France is
  disproportionately weighted in private, a healthy public score could be
  actively misleading — and France is already the part that cannot be validated
  locally. Those two unknowns compound.

Working assumption until told otherwise: treat the public score as a weak,
noisy signal. Prefer local CV with leave-one-country-out.

---

## No test labels — validation carries most of the signal

Chunk 4: test ground truth is not provided, and the PS instructs holding out a
validation split from train and self-scoring.

### ⚠️ Correction to chunk-4 inference

From chunk 4 I inferred **little or no leaderboard feedback loop.**
**That was an over-reach.** Chunk 5 states there *is* a leaderboard and that
`matching_results.tsv` is uploaded to the Portal during the challenge.

The accurate version: a leaderboard exists and returns a score, but chunk 5's
phrase *"catch a rejection locally instead of spending a submission on it"*
implies **submissions are limited.** So feedback exists but cannot be used to
probe freely.

Revised consequence — weaker than what I wrote, still the same direction:

> Local validation carries most of the signal. Leaderboard attempts are a
> scarce resource, not a search tool. A holdout that doesn't reflect test will
> waste attempts you can't get back.

Validation design remains the highest-leverage artefact in the build — not the
model.

### The France problem has no clean validation answer

France appears only in test. So **local validation cannot measure the thing
most likely to break.** There is no French data to hold out.

The only available proxy is leave-one-country-out: train on US, validate on
India (and vice versa) to simulate a shift to an unseen country. It measures
*generalisation to an unseen country* rather than *generalisation to France
specifically* — an imperfect proxy, but the only one the data permits.

Worth stating plainly: the French subset of the test set is, and will remain,
unmeasurable before submission.

---

## Schema — confirmed (chunk 3)

Every source file has **exactly four columns**:

| Column | Notes |
|---|---|
| `entity_id` | Unique per record. Prefix gives the source: `S1-`, `S2-`, `S3-` |
| `business_name` | Abbreviations, legal suffixes, typos, transliterations |
| `business_address` | Partial addresses, format variations, missing components, landmark references |
| `country` | **Open set of string labels.** Train: US, India. Test adds **France** |

There is no `source` column — source comes from the `entity_id` prefix and from
which file the row sits in.

### Ground truth: `train_ground_truth.tsv`

| Column | Notes |
|---|---|
| `source1_entity_id` | An S1 record's id |
| `matched_entity_ids` | Comma-separated ids from S2 **and/or** S3. **Empty when no matches** |

### ⚠️ Correction to chunk-2 inference

From chunk 2 I inferred **two** list columns (`source2_ids`, `source3_ids`).
**That was wrong.** It is **one** column, `matched_entity_ids`, holding S2 and
S3 ids mixed together. Source is recoverable from each id's prefix.

Consequences of the real format:
- Output is **one flat set per S1 entity**, not two per-source sets.
- S2 and S3 are therefore **scored together**, which partly resolves the
  earlier open question about joint vs separate scoring.
- Only S1-anchored pairs matter. S2↔S3 linkage is not part of the output.

### Two hard submission constraints

1. **Zero-match is encoded as empty** — the row still exists, the list field is
   empty. Dropping empty rows loses score silently.
2. **Every test entity must appear in the submission**, France included. So
   submission row count must equal the test S1 entity count, exactly.

---

## Output format — two files, and the second one is an architectural constraint

Confirmed in chunk 5. Both land in `output/`.

### 1. `output/matching_results.tsv` — scored

| Column | Description |
|---|---|
| `source1_entity_id` | An S1 record's id |
| `matched_entity_ids` | Comma-separated S2/S3 ids |

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

Note `S1-00003` — present, list empty. Also note the **id format**:
`S{n}-` prefix plus a zero-padded 5-digit number.

### 2. `output/candidate_pairs.tsv` — not scored, but binding

| Column | Description |
|---|---|
| `source1_entity_id` | An S1 record's id |
| `candidate_entity_ids` | Comma-separated candidate S2/S3 ids |

**This is the constraint that shapes the pipeline.** The PS is unusually
specific about what belongs here:

> the exact set of records you feed into your matching model for inference —
> the final candidate list just before the ML model scores them, not the raw
> output of an early blocking pass you later filter further. If your pipeline
> has several blocking/filtering stages, `candidate_pairs.tsv` is the last one:
> whatever your model actually runs inference over.

Read carefully, this mandates a design, not just a file:

- The pipeline must have a **clean, materialisable split** between candidate
  generation and model scoring. The boundary has to be a concrete artefact.
- **`matching_results` ⊆ `candidate_pairs`** is enforced; the validator warns
  on violation. A matched id that was never a candidate is treated as a bug.
- They state they use it to measure **recall ceiling and reduction ratio** —
  the exact quantity flagged in chunk 1 as where ER is won or lost. They are
  measuring it directly.
- Unscored on the leaderboard ≠ irrelevant. It is used to *verify your
  pipeline*, so it plausibly feeds qualitative or final judging. Treat it as a
  real deliverable, not a formality.

### Hard rules for both files

| Rule | Failure mode if violated |
|---|---|
| Exactly one row per S1 test entity | Rejection |
| Empty list for singletons — row still present | Silent score loss or rejection |
| No duplicate ids within a single list | Rejection |
| Only S2-/S3- ids **that exist in the test set** | Rejection |
| Tab separated, **no quoting** | Rejection / zero score |
| `matching_results` ⊆ `candidate_pairs` | Validator warning, signals a bug |

### Validator — use it every time

`utils/validate_submission.py`, stdlib only, run from `student_resource/`:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

`PASS` with exit 0, or a numbered issue list with exit 1. It checks format
only — **it does not compute the score.** Its existence plus the phrase
"spending a submission" is the strongest evidence yet that **submissions are
limited.**

### Directory layout — now resolved

```
student_resource/
├── dataset/
│   ├── train/  train_source{1,2,3}.tsv, train_ground_truth.tsv
│   └── test/   test_source{1,2,3}.tsv
├── output/     matching_results.tsv, candidate_pairs.tsv
└── utils/      validate_submission.py
```

`student_resource/` is the working root — the validator expects to be run from
there. A "Final Submission Package" section is referenced and has not arrived.

---

## Fair play: external data lookup is banned — disqualification risk

Chunk 9, and it is unambiguous. **Strictly prohibited:**

- Commercial entity-resolution APIs or services
- Government business-registration lookups
- **Geocoding APIs to normalise addresses** — named explicitly
- Any external data augmentation from internet sources

Enforcement: all approaches, methodologies and code pipelines are reviewed;
evidence of external lookup means **immediate disqualification**. The stated
intent — *"using only the provided training data."*

### The boundary that matters: models vs. data

Chunks 6 and 9 are consistent, but the line between them has to be held
deliberately because getting it wrong is disqualifying, not just costly.

| | Status | Source |
|---|---|---|
| Pretrained model weights, MIT/Apache, ≤8B, run locally | **Allowed** | Chunk 6, explicit |
| Geocoding API (Nominatim, Google Maps, etc.) | **Banned** | Chunk 9, named |
| Government / commercial registry lookup | **Banned** | Chunk 9, named |
| Downloaded gazetteer or business dataset | **Banned** | "external data augmentation from internet sources" |
| Hand-authored normalisation rules (`Rd`→`Road`, `Pvt`→`Private`) | **Judgement — reads as allowed** | Domain knowledge in code; resolves no identity |

The rule targets *looking up business identities or resolving entities*. A
pretrained encoder is a model, not a lookup, and chunk 6 permits it explicitly.
A hand-written abbreviation map encodes domain knowledge and identifies no
specific business.

A **downloaded** gazetteer — a city list, a postcode table — is the genuinely
risky case. It is external data from the internet even though it resolves no
business directly. Given the penalty is disqualification rather than a score
hit, the asymmetry argues for staying clearly inside the line and documenting
every external artefact in the methodology write-up.

---

## Organiser tips — what they reveal

Chunk 9's tips are worth reading as signal about expected solutions, not as
filler.

| Tip | What it tells us |
|---|---|
| "Invest in a strong blocking/candidate generation strategy — it determines the upper bound of your recall" | **Confirms the chunk-1 structural read**, now stated outright by the organisers. Third independent signal that blocking is first-class. |
| "Explore string similarity features (Jaccard, Levenshtein, TF-IDF cosine)" | They point at **classical string similarity**, not embeddings or transformers. Notable given chunk 6 permits ≤8B models — the organisers expect classical features to carry real weight. |
| "Pay attention to country-specific address patterns" | See the tension below. |
| "F_0.5 rewards precision more than recall" | Restates the metric bias. |
| "Do not neglect singletons — correctly predicting 'no match' is worth a full 1.0" | **Confirms the singleton read.** They are telling you directly where the free score is. |
| "Validate your own output format before submitting" | Use the provided validator as a gate. |

### The tension worth naming

Chunk 3 says: *do not hard-code, filter, or one-hot your pipeline to only
{US, India}.*
Chunk 9 says: *pay attention to country-specific address patterns.*

These pull opposite ways, and the reconciliation matters because France is in
test only. Reading them together: **country-conditional behaviour is fine, but
it must degrade gracefully to an unseen country** rather than fail or fall
through to nothing. Anything with a hard branch per country needs a sane
default path for a label never seen in training.

---

## Model constraint: MIT / Apache 2.0, ≤ 8B parameters

Chunk 6, verbatim: *"Final model should be a MIT/Apache 2.0 License model and up
to 8 Billion parameters."*

This resolves the open question about external pretrained models — **they are
allowed**, within two limits. And chunk 6 states the packages of top teams are
**reviewed in detail** for "fair-play and model-license rules" before final
rankings, so this is audited, not self-certified.

### What the licence clause excludes

The restriction to MIT / Apache 2.0 is narrower than "open weights". Families
that do **not** qualify under a plain reading:

- **Llama** (all versions) — Meta community licence, not MIT/Apache.
- **Gemma** — Google custom terms.
- Anything under a bespoke "open but restricted" or non-commercial licence.

Families that generally **do** qualify and are relevant here — multilingual
sentence encoders, which matter directly because of France:

| Model family | Licence | Rough size |
|---|---|---|
| LaBSE | Apache 2.0 | ~470M |
| multilingual-E5 (small/base/large) | MIT | 118M – 560M |
| XLM-RoBERTa (base/large) | MIT | 270M – 560M |
| BGE-M3 | MIT | ~570M |
| paraphrase-multilingual-MiniLM | Apache 2.0 | ~118M |
| Mistral 7B / Qwen (Apache-licensed variants) | Apache 2.0 | ~7B |

Licences must be verified per checkpoint at the time of use, not assumed from
the family name — variants within a family sometimes differ.

### Two observations about the cap

1. **8B is generous relative to what this task needs.** ER over a business name
   and an address is a short-text similarity problem. Multilingual encoders in
   the 100M–600M range are the natural fit and all sit far below the cap. The
   constraint is unlikely to bind unless the approach reaches for an LLM it
   doesn't need — which, on a 2-4 day clock with GPU quota still pending, would
   be a poor trade regardless of the rules.

2. **Licence choice is cheap now and expensive later.** Because packages are
   audited before final rankings, a licence-incompatible checkpoint discovered
   late means rebuilding the scorer. Pick clean from the start and record the
   licence in the write-up.

### Ambiguity worth resolving

"Final **model**" is singular. It does not say whether the constraint covers
only the matching model or every model in the pipeline, including any encoder
used for blocking. Conservative reading: **all of them.** No downside to
complying everywhere, since the compliant options are also the sensible ones.

---

## Final submission package

Separate from the live leaderboard uploads, a single zip is submitted:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv      # same file uploaded to the leaderboard
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                  # all source
│       ├── README.md             # end-to-end reproduction steps
│       └── requirements.txt      # pinned dependencies
└── Documentation_template.md     # filled-in methodology write-up
```

**It is used to reproduce results, audit blocking, and check licence
compliance.** Top teams' packages get detailed review before rankings are
confirmed.

Implications:

- **Reproducibility is graded, not assumed.** "Anyone should be able to
  regenerate both output files ... using only what is in this folder." That
  rules out notebook-only workflows and undocumented manual steps.
- **`requirements.txt` must pin versions.** Already have `requirements-lock.txt`
  from Phase 0, so this is nearly free.
- **`Documentation_template.md` is provided and filled in, not rewritten.** Do
  not rename it. Its contents haven't been shown yet, but chunk 8 gives the
  required sections:
  - Methodology used
  - **Candidate generation / blocking strategy**
  - Model architecture and feature engineering
  - Any other relevant information

  **No page limit — "prioritise clarity and technical depth over brevity."**
  That phrasing, plus blocking getting its own mandated section and
  `candidate_pairs.tsv` being audited, means blocking is treated as a
  first-class part of the evaluation rather than plumbing. Write it up
  properly, with the recall-ceiling and reduction-ratio numbers.
- The repo layout should mirror `code/business_entity_resolution/src/` from the
  start, so packaging is a copy rather than a reorganisation under deadline.

### Additional rejection rules from chunk 6

Beyond chunk 5's list:

| Rule | Note |
|---|---|
| **No self-matches to Source 1** | `S1-` ids in `matched_entity_ids` are rejected. New, explicit. |
| No ids absent from the test set | Restated |
| Every Source 1 entity must appear | Restated — missing entities cause rejection |
| No duplicate `source1_entity_id` **rows** | New — duplicates at row level, not just within lists |
| Failed validation → **not evaluated** | A `SCORED` status with an F₀.₅ value is the success signal |

---

## The only three signal fields

`business_name`, `business_address`, `country`. That is the entire feature
space.

No phone. No URL. No category. No coordinates. No registration number.

Everything is fuzzy comparison over two free-text fields, partitioned by a
country label. The tiny field count is itself a structural fact: there is
nowhere to hide, and feature engineering has a hard ceiling set by what can be
extracted from a name and an address.

---

## France — a deliberate generalisation trap

Train covers US and India. Test adds **France**, absent from training. The PS
calls this out explicitly and unusually directly:

> Treat country as an open set of string labels: do not hard-code, filter, or
> one-hot your pipeline to only {US, India}, and remember that every test
> entity — France included — must appear in your submission.

Reading this as spec, not suggestion. Implications:

- **One-hot encoding `country` breaks at test time.** Explicitly named. An
  unseen category yields an all-zero row or a crash depending on the encoder.
- **Country-specific parsing breaks.** Indian PIN codes and US state
  abbreviations are useless for French addresses. Anything keyed to a country's
  address grammar has no French branch.
- **Legal-suffix knowledge does not transfer.** US/India training teaches
  `Corp`, `Inc`, `Pvt`, `Ltd`. France uses `SARL`, `SA`, `SAS`, `EURL` — none
  of which appear in training data.
- **French orthography is new** — diacritics (`é`, `è`, `ç`), different
  tokenisation conventions.
- Whatever handles name/address similarity must be **language and script
  agnostic**, because there is zero French training signal.

The explicit warning is informative in itself: **they warn about the things
that actually sink people.** The naive pipeline fails here, by design.

---

## What the noise spec tells us about comparison semantics

The PS hands over the noise taxonomy directly, which is a gift — it's
effectively a normalisation checklist.

**Name variations:**
abbreviations (`Corp`/`Corporation`, `Pvt`/`Private`, `Ltd`/`Limited`), legal
suffix inconsistencies, DBA/trade names, punctuation (`&` vs `and`),
**word-order transpositions**, typos.

**Address variations:**
abbreviations (`Rd`/`Road`, `St`/`Street`), transliteration variants, missing
components (no PIN, no state), landmark references (`Near SBI ATM`), municipal
numbering formats, **component reordering**.

Two of these are structurally decisive rather than cosmetic:

1. **Word-order transposition and component reordering are named explicitly.**
   That means comparison must be **order-insensitive**. Any sequence-sensitive
   similarity degrades on exactly the noise the PS promises will be present.

2. **Missing components are named explicitly.** So similarity must degrade
   gracefully on absent fields rather than treating absence as mismatch. An
   address with no PIN code is still the same address.

3. **DBA/trade names** are the nastiest item on the list — a legitimate match
   where the two names may share almost no tokens. No amount of string
   normalisation solves that case; it is a genuine ceiling on recall from
   name similarity alone.

---

## Data & I/O — confirmed

- **TSV throughout.** Tab-separated in, tab-separated out.
- **Layout:** `dataset/{split}/{split}_source{n}.tsv`, plus
  `dataset/train/train_ground_truth.tsv`.
- **`train/` split confirmed.** A `test/` split follows from the PS text.

### I/O traps

| Trap | Consequence |
|---|---|
| Missing `sep="\t"` on read | Silently one column holding the whole line. The PS warns about this explicitly. |
| IDs read as non-string | `entity_id` is prefixed (`S1-…`) so it stays string — but ids inside `matched_entity_ids` must survive split/strip intact. |
| Default pandas quoting | Business names carry quotes and apostrophes; a stray `"` can swallow rows. |
| Embedded tabs in dirty text | Column shift. Assert field count on load, don't trust it. |
| Empty list read as `NaN` | `matched_entity_ids` is legitimately empty. `NaN` vs `""` must be handled deliberately, both on read and write. |
| Dropping zero-match rows | Direct score loss. Every test entity must appear. |
| Submission written without `sep="\t"` | Scores zero — comma default breaks on addresses. |
| Unicode normalisation | French diacritics and Indian transliterations need a deliberate NFC/NFKD decision, applied identically to both sides of every comparison. |

---

## Signal table

| Category | Status after chunks 1-3 |
|---|---|
| Task type | ER / pairwise linkage, 0..N cardinality |
| Sources | 3. S1 deduplicated reference; S2, S3 linked against it |
| Direction | S1 → S2 and S1 → S3, scored as one combined set |
| Fields | `entity_id`, `business_name`, `business_address`, `country` — that's all |
| Modality | Two free-text fields plus one open-set categorical |
| File format | TSV in and out |
| Submission unit | One row per S1 entity, single comma-separated id list |
| Zero-match encoding | **Empty field** — row still present |
| Row coverage | **Every test entity must appear**, France included |
| Train split | Confirmed, with `train_ground_truth.tsv` |
| Labelled match pairs | **CONFIRMED** — ground truth provides them |
| Distribution shift | **CONFIRMED** — France in test only |
| Evaluation metric | **F₀.₅ — precision-weighted 4:1.** Exact formula pending |
| Micro vs macro F₀.₅ | **UNKNOWN — changes optimal behaviour substantially** |
| Test labels | **None.** Self-scored holdout needed for local measurement |
| Leaderboard | **EXISTS** — upload `matching_results.tsv` to the Portal |
| Public LB | Subset of test, live feedback during the challenge |
| Private LB | Remaining test portion; **final rankings come from this** |
| Split mechanics | Predict the **full** test set; split applied at scoring |
| Public/private ratio | **UNKNOWN** |
| Split stratified by country? | **UNKNOWN — compounds the France blind spot** |
| Test files | All three sources present (`test_source{1,2,3}.tsv`) |
| Output files | **Two:** `matching_results.tsv` (scored), `candidate_pairs.tsv` (analysed) |
| Pipeline shape | **Mandated two-stage** — candidate gen must be a materialised artefact |
| Subset constraint | `matching_results` ⊆ `candidate_pairs`, validator-enforced |
| Validator | `utils/validate_submission.py`, stdlib, format only, no score |
| Working root | `student_resource/` |
| ID format | `S{n}-` + zero-padded 5 digits, e.g. `S1-00001` |
| Pretrained models allowed | **YES — MIT/Apache 2.0 only, ≤ 8B params.** Audited |
| Deliverables | Leaderboard uploads **+** a final zip (code, outputs, methodology) |
| Reproducibility | **Graded.** Package must regenerate both outputs standalone |
| Self-matches | **Forbidden** — no `S1-` ids in `matched_entity_ids` |
| F₀.₅ exact formula | **CONFIRMED** — `1.25·P·R / (0.25·P + R)` |
| Micro vs macro | **MACRO** — per S1 entity, then averaged |
| Singleton scoring | **1.0 if correctly empty, 0.0 if any prediction.** Included in average |
| Row counts / data size | **UNKNOWN** |
| Submission limits | **Implied limited** ("spending a submission") — exact count UNKNOWN |
| Hard deadline | **UNKNOWN** |
| AWS scored | **UNKNOWN** — no mention in any chunk so far |
| External data / APIs | **BANNED.** Geocoding, registries, augmentation → disqualification |
| Pretrained weights | **Allowed** (chunk 6) — a model is not a lookup |

---

## Open questions

Ranked by how much each changes the build.

1. **What fraction of S1 entities are singletons?** Now the most valuable
   unknown. Under macro scoring with singletons at 1.0/0.0, that fraction is
   directly the share of total score obtainable from abstention alone.
   Answerable from `train_ground_truth.tsv` the moment the data is in hand.

2. **Does the ≤8B / licence constraint apply to every model in the pipeline, or
   only the final matcher?** "Final model" is singular. Conservative reading is
   all of them; worth confirming since it costs nothing to comply.

3. **Row counts per source.** Decides whether this runs on the M1 or needs the
   pending GPU quota, and whether the quadratic candidate space is merely large
   or genuinely intractable.

4. **Is `country` guaranteed present and consistent?** If it can be null or
   disagree across sources for the same business, it cannot be trusted as a
   hard partition.

5. **Can a match cross countries?** Multinational businesses would break a
   country-partitioned approach. Not addressed by the PS so far.

6. **How many leaderboard submissions do we get?** Chunk 5 implies limited
   ("spending a submission") without giving a number. Determines how much of
   the threshold search can be done on the leaderboard versus locally.

7. **Hard deadline** — still unstated.

8. ~~**Is `candidate_pairs.tsv` judged, even if unscored?**~~ → **RESOLVED, and
   it changes strategy.** See "Candidate-set size is a ranking criterion" below.

---

## ⚠️ Candidate-set size is a ranking criterion (PS update banner)

Added to the top of the live problem statement, and **not present in the nine
chunks logged below**. Captured 2026-09-25 from a screenshot of the portal:

> 📣 **Update: `candidate_pairs.tsv` is part of your final submission**
>
> 1. **Blocking has to scale.** Amazon resolves business entities across
>    billions of records, so comparing every record with every other one is not
>    an option. Your blocking / candidate-generation step must cut the search
>    space to a small candidate set per Source 1 entity.
> 2. **Candidate generation counts toward the final ranking.** We will review
>    your `candidate_pairs.tsv` and the code that produces it when deciding
>    final rankings, alongside your `matching_results.tsv` score. **The approach
>    that generates a smaller candidate set per Source 1 entity will be ranked
>    higher** in the final evaluation beyond the public/private leaderboard.

### What this changes

Open question 8 above assumed the reduction ratio might be cosmetic. It is not.
There are **two** ranked axes, not one:

| Axis | Measured by | Direction |
|---|---|---|
| Match quality | macro F₀.₅ on the private split | maximise |
| Blocking parsimony | mean candidates per Source 1 entity | **minimise** |

The second axis is a tiebreak applied *after* the leaderboard, so score still
dominates — but among teams clustered at similar scores (and the top 500 are all
above 0.95, i.e. **very** clustered), candidate volume is what separates them.

### Where the v2 config actually stands

The v2 rebuild was chosen on recall ceiling alone, before this rule was known.
Checked against `pipeline.BlockingConfig` rather than assumed, it turns out to
sit in a defensible place on both axes — but for the wrong reason (compute cost,
not parsimony):

| config | US ceiling | blocking time | mean cand / S1 |
|---|---|---|---|
| v1 — `top_n=40`, `max_df=0.01`, + exact | 0.9531 | 376 s | **76.8** (measured) |
| **v2 (running) — `top_n=40`, `max_df=0.50`, + exact** | ~0.98 | ~600 s | **?** — must measure |
| rejected — `top_n=100`, `max_df=0.50`, + exact | 0.9825 | 1412 s | ~2x v2 |

`top_n` stayed at 40. The recall came almost entirely from relaxing `max_df`
(0.01 -> 0.50), which is a *pruning* change, not a *fan-out* change: it alters
which n-grams carry weight, not how many neighbours are kept per entity. So the
candidate count per S1 entity is bounded by the same `top_n=40` union as v1.

**This is the key structural fact.** `top_n` sets candidate volume; `max_df`
sets candidate quality. v2 bought recall on the quality axis and left the volume
axis untouched. Under the new ranking rule that is the right trade, arrived at
by luck rather than design.

### What still has to be measured

Volume is bounded by `top_n`, but not equal to it — the union of two top-40
lists plus exact-key groups produced a mean of 76.8 in v1, and `max_df=0.50`
changes which neighbours clear `min_sim`, so v2's mean will differ. **Measure
the v2 mean the moment the shards land**, before touching anything else.

### The prune stage, if the number comes back high

The PS defines `candidate_pairs.tsv` as *"the exact set of records you feed into
your matching model for inference"* and states: **"If your pipeline has several
blocking/filtering stages, `candidate_pairs.tsv` is the last one."**

Wide blocking is therefore not penalised — only a wide *final* set is:

```
wide recall-greedy blocking  ->  cheap high-recall prune  ->  candidate_pairs.tsv  ->  model
```

The prune must be cheap (no model inference) and near-lossless on recall. It
cuts the audited artefact and real inference cost together. This is the
structure the PS describes, not a loophole. Only build it if the measured mean
justifies it.

### Resolved by chunk 9

- ~~Is external *training data* allowed?~~ → **no.** External databases, APIs,
  geocoding services and internet data augmentation are all banned, enforced by
  code review, penalty is disqualification.
- ~~Is blocking actually important or am I over-reading it?~~ → organisers
  state it outright: it "determines the upper bound of your recall."
- ~~Are singletons worth explicit effort?~~ → organisers say so directly.

### Resolved by chunk 8

- ~~Is leaderboard feedback trustworthy for tuning?~~ → public/private split
  confirmed; final ranking is private. Public is a weak signal, not a target.
- ~~What must the methodology document cover?~~ → methodology, blocking
  strategy, model architecture and feature engineering, plus anything else
  relevant. No page limit.

### Resolved by chunk 7

- ~~Exact F₀.₅ formula~~ → `(1.25 × P × R) / (0.25 × P + R)`.
- ~~Micro or macro?~~ → **macro**, per S1 entity then averaged.
- ~~How are singletons treated?~~ → included; 1.0 for a correct empty
  prediction, 0.0 for any prediction. Correcting my earlier "1:4" weighting
  claim: the PS states 2×, which is the conventional reading.

### Resolved by chunk 6

- ~~Are pretrained models allowed?~~ → **yes**, MIT/Apache 2.0, ≤ 8B params,
  and audited for the top teams.
- ~~What else is submitted besides the leaderboard file?~~ → a zip with code,
  pinned requirements, both outputs, and a filled-in methodology template.
- ~~Is reproducibility graded?~~ → yes, explicitly.

### Resolved by chunk 5

- ~~Exact submission header~~ → `source1_entity_id`, `matched_entity_ids`.
- ~~Zero-match row handling~~ → row present, list empty, exactly one row per
  test S1 entity.
- ~~Is there a leaderboard?~~ → **yes.** Corrects the chunk-4 over-reach.
- ~~Directory layout~~ → `student_resource/{dataset,output,utils}`.
- ~~ID format~~ → `S{n}-` + zero-padded 5 digits.

### Resolved by chunk 4

- ~~Evaluation metric~~ → **F₀.₅**, precision-weighted. Formula pending.
- ~~Are test labels provided?~~ → no. Self-scored holdout is the only signal.
- ~~Do all three sources exist in test?~~ → yes.

### Resolved by chunk 3

- ~~Field schema~~ → four columns, confirmed.
- ~~Labelled match pairs exist?~~ → yes, `train_ground_truth.tsv`.
- ~~Zero-match encoding~~ → empty field, row retained.
- ~~S2/S3 scored jointly or separately~~ → jointly, one flat set.
- ~~Two list columns or one~~ → one. Earlier inference was wrong.

---

## Chunk log

### Chunk 1 — raw

> In large-scale commercial platforms, business identity data arrives from
> multiple independent sources — each contributing partial, noisy fragments of
> information about the same real-world entities. These fragments share no
> common identifiers, and the challenge of determining which records refer to
> the same business is known as Entity Resolution (ER). Your challenge is to
> build an ML solution that, given business records from 3 independent data
> sources with noisy and inconsistent fields, determines which records across
> sources refer to the same real-world business entity.
>
> Source 1 is the deduplicated reference source. Your task is to find all
> matching records from Source 2 and Source 3 for each Source 1 entity. A
> Source 1 entity may match zero, one, or many records from Source 2 and
> Source 3.

**Read:** task family, S1-anchored direction, 0..N cardinality, no shared keys.

### Chunk 2 — raw

> **File Format.** All files in this challenge are tab-separated (.tsv), and
> your submissions must be tab-separated too. Tabs are used because business
> addresses and the ID list columns both contain commas. Read them with an
> explicit tab separator, for example:
> `df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")`
> Reading a .tsv without `sep="\t"` will silently produce a single column
> containing the whole line.

**Read:** TSV mandatory both directions; directory layout and naming;
train split exists. Inferred two list columns — **later corrected to one**.

### Chunk 3 — raw

> **Data Description.** Each source file has columns: `entity_id` (prefix
> indicates source S1-/S2-/S3-), `business_name` (abbreviations, legal
> suffixes, typos, transliterations), `business_address` (partial addresses,
> format variations, missing components, landmark-based references), `country`
> (training covers US and India; test additionally contains France, absent from
> training — treat country as an open set, do not hard-code, filter or one-hot
> to {US, India}; every test entity including France must appear in the
> submission). No separate source column.
>
> `train_ground_truth.tsv` has `source1_entity_id` and `matched_entity_ids`
> (comma-separated matching ids from Source 2 and/or Source 3, empty when the
> entity has no matches).
>
> **Noise patterns:** name — abbreviations (Corp/Corporation, Pvt/Private,
> Ltd/Limited), legal suffix inconsistencies, DBA/trade names, punctuation
> (& vs and), word-order transpositions, typos. Address — abbreviations
> (Rd/Road, St/Street), transliteration variants, missing components (no PIN,
> no state), landmark references (Near SBI ATM), municipal numbering formats,
> component reordering.

**Read:** the whole schema, and three things that reshape the problem — the
single mixed id list (correcting chunk 2), the France distribution shift with
an explicit warning against one-hot/filtering, and a noise taxonomy that makes
order-insensitive comparison and graceful handling of missing components
structural requirements rather than refinements. Also confirmed supervised
labels exist, and that the entire signal is three fields.

### Chunk 4 — raw

> **Dataset Details.** Training dataset: business records across 3 sources with
> ground truth matching labels. Test set: business records across 3 sources
> without matching labels.
>
> Training files: `dataset/train/train_source1.tsv` (the deduplicated reference
> source), `train_source2.tsv`, `train_source3.tsv`,
> `train_ground_truth.tsv`.
>
> Test files: `dataset/test/test_source1.tsv` — generate matches for every
> entity in this file — plus `test_source2.tsv`, `test_source3.tsv`.
>
> No ground truth is provided for the test set. To measure your own
> performance, hold out a validation split from the training data and score it
> yourself using the F_0.5 formula given below.

**Read:** the metric is **F₀.₅**, precision-weighted roughly 4:1, which
inverts the usual ER bias at the decision stage while leaving the blocking
recall argument intact. No test labels and an instruction to self-score imply
little or no leaderboard feedback, which promotes validation design to the
highest-leverage artefact in the build. Combined with France being absent from
train, the most likely failure mode is also the one that cannot be measured
locally — leave-one-country-out is the only available proxy. Row coverage
requirement restated ("every entity in this file").

### Chunk 5 — raw

> **Output Format.** Two tab-separated files in `output/`:
> `matching_results.tsv` — final entity matches, the only file scored on the
> leaderboard, uploaded to the Portal during the challenge; and
> `candidate_pairs.tsv` — the candidate set your blocking stage produced,
> before your final matching model narrowed it down.
>
> `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`
> (comma-separated S2/S3 ids). Every Source 1 entity in the test set must have
> exactly one row; leave `matched_entity_ids` empty for singletons; no
> duplicate ids within a list; ids must only be S2/S3 ids that exist in the
> test set.
>
> `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`. This is
> the exact set fed to the matching model for inference — the last
> blocking/filtering stage, whatever the model actually runs inference over,
> not an early pass later filtered. Every id in `matching_results.tsv` should
> appear here. Not scored; used to analyse blocking quality (recall ceiling,
> reduction ratio) and verify the pipeline. Final matches should be a subset of
> candidates; a matched id that never appeared as a candidate signals a
> pipeline bug and the validator warns about it.
>
> Validate before submitting with `utils/validate_submission.py` (stdlib only),
> run from `student_resource/`, passing `--matching`, `--candidate` and
> `--test-dir dataset/test`. Prints PASS (exit 0) or a numbered issue list
> (exit 1). It reads only the output files and test sources; it does not
> compute the score.

**Read:** three substantive things. First, a leaderboard exists — correcting my
chunk-4 over-reach — but "spending a submission" implies attempts are limited.
Second, `candidate_pairs.tsv` is not a reporting artefact, it is an
**architectural mandate**: the boundary between candidate generation and model
scoring must be a concrete materialised set, with `matching_results` a strict
subset of it. Third, they explicitly measure recall ceiling and reduction ratio
— the exact quantity called out in chunk 1 as where ER is won or lost, which
confirms the read rather than leaving it as inference. Also resolved: exact
headers, id format, directory layout, and a stdlib validator that removes any
excuse for a format-based rejection.

### Chunk 6 — raw

> **Final Submission Package.** In addition to live leaderboard uploads, every
> team submits a single zip with code and outputs. Used to reproduce results,
> audit blocking, and check fair-play and model-licence rules — the top teams'
> packages are reviewed in detail before final rankings are confirmed.
>
> Structure: `<team_name>_submission.zip` containing `output/`
> (`matching_results.tsv`, `candidate_pairs.tsv`),
> `code/business_entity_resolution/` (`src/`, `README.md` with end-to-end
> reproduction steps, `requirements.txt` pinning versions), and
> `Documentation_template.md` (the filled-in methodology template; .md or .pdf,
> no need to rename). Anyone should be able to regenerate both output files from
> the training/test data using only what is in that folder.
>
> Constraints: (1) format output exactly as described — submissions failing
> validation are not evaluated; a `SCORED` status with an F_0.5 score indicates
> correct formatting. (2) `matched_entity_ids` must only reference Source 2 or
> Source 3 entities — self-matches to Source 1 and ids not in the test set are
> rejected. (3) Every Source 1 entity must appear; missing entities cause
> rejection. (4) Duplicate entity ids in any list cause rejection, as do
> duplicate `source1_entity_id` rows. (5) **Final model should be a MIT/Apache
> 2.0 License model and up to 8 Billion parameters.**

**Read:** the licence-and-size clause is the significant item. Pretrained models
are permitted, which gives the France problem a viable tool in multilingual
sentence encoders — but MIT/Apache 2.0 excludes Llama and Gemma, and compliance
is audited before final rankings rather than taken on trust. The 8B cap is
generous relative to a short-text similarity task, so it is unlikely to bind
unless the approach over-reaches. Separately, reproducibility is now a graded
deliverable: the package must regenerate both outputs standalone, which rules
out notebook-only workflows and argues for mirroring
`code/business_entity_resolution/src/` in the repo layout from the start rather
than reorganising under deadline. Two new rejection rules appeared —
no S1 self-matches, and no duplicate `source1_entity_id` rows.

### Chunk 7 — raw

> **Evaluation Criteria.** Submissions are evaluated using F_β Score (β = 0.5) —
> a precision-heavy metric that penalizes false merges (matching two different
> businesses) more than missed matches.
>
> `F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
>
> Computed as a **macro-average**: F_0.5 is calculated per Source 1 entity, then
> averaged across all Source 1 entities in the evaluation set.
>
> Singletons are included in that average. A Source 1 entity with no true
> matches scores 1.0 when you correctly predict an empty list, and 0.0 when you
> predict any match for it. Correctly identifying singletons therefore earns
> credit, and false merges on them are penalised.
>
> Why precision-heavy? In real-world entity resolution, merging two distinct
> businesses (false positive) is more damaging than missing a link (false
> negative). F_0.5 weights precision 2× over recall.
>
> Example: predicted `[S2-00047, S2-00193, S3-00812]`, truth
> `[S2-00047, S3-00812]` → Precision = 2/3, Recall = 1.0,
> F_0.5 = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) = 0.714

**Read:** the two decisive facts are **macro-averaging** and the **singleton
rule**. Macro means every S1 entity carries equal weight regardless of match
count, so there is no advantage in doing well on high-cardinality entities.
The singleton rule is all-or-nothing — 1.0 for a correct empty prediction, 0.0
for any prediction at all — which makes the singleton share of S1 directly
equal to the score obtainable from abstention alone, before any matching skill.
Worked through, the per-entity arithmetic shows that missing a true match
(0.833 on a 2-match entity) beats catching both while adding one false positive
(0.714), and that a single spurious id on a 1-match entity nearly halves its
score (1.0 → 0.556). This makes the abstain decision the largest single lever in
the build, ahead of the scorer. Also corrected my earlier "1:4 in precision's
favour" phrasing — the PS says 2×, which is the conventional F_β reading; the
4:1 figure describes the harmonic-mean weights, not the standard statement.
PS example verified independently: 0.8333 / 1.16667 = 0.7143. ✓

### Chunk 8 — raw

> **Leaderboard Information.** Public Leaderboard: during the challenge,
> rankings are based on a subset of the test set, giving real-time feedback.
> Private Leaderboard: after the challenge ends, the private leaderboard is
> revealed, using the remaining portion of the test set. Final Rankings: the
> final decision is based on the private leaderboard. You submit predictions for
> the full test set in both cases; the split is applied during scoring.
>
> **Submission Requirements.** (1) Leaderboard during the challenge: upload
> `matching_results.tsv` in the Portal — tab-separated, exact column names.
> This drives both public and private leaderboards. (2) Final submission
> package: the single zip described above; all teams must submit it, and the
> top teams' packages are reviewed before final rankings are confirmed.
> (3) The methodology document must describe: methodology used; candidate
> generation/blocking strategy; model architecture and feature engineering; any
> other relevant information. A template is provided in
> `Documentation_template.md`. **There is no page limit — prioritise clarity and
> technical depth over brevity.**

**Read:** the public/private split is the standard overfitting trap, and two
features of this competition sharpen it. The metric is macro-averaged, which
carries higher variance on a subset than a pooled metric would, so the public
score is noisier than it looks. And the abstain threshold is already identified
as the largest lever in the build — tuning the highest-leverage parameter
against the noisiest signal is the classic way to lead publicly and lose
privately. Neither the split ratio nor whether it is stratified by country is
stated; if France is weighted differently across the two, a healthy public
score becomes actively misleading, and France is already the segment that
cannot be validated locally. Separately, blocking now has a mandated section in
the methodology document *and* an audited artefact in `candidate_pairs.tsv` —
consistent signals that it is treated as first-class, not plumbing.

### Chunk 9 — raw

> **Academic Integrity and Fair Play.** ⚠️ STRICTLY PROHIBITED: External Data
> Lookup. Participants are STRICTLY NOT ALLOWED to use external databases, APIs,
> or services to look up business identities or resolve entities. This includes
> but is not limited to: commercial entity resolution APIs or services; looking
> up business registrations from government databases; using geocoding APIs to
> normalize addresses; any external data augmentation from internet sources.
>
> Enforcement: all submitted approaches, methodologies, and code pipelines will
> be thoroughly reviewed and verified. Any evidence of external data lookup will
> result in immediate disqualification. Fair Play: this challenge is designed to
> test your machine learning and data science skills using only the provided
> training data.
>
> **Tips for Success:** invest in a strong blocking/candidate generation strategy
> — it determines the upper bound of your recall; explore string similarity
> features (Jaccard, Levenshtein, TF-IDF cosine) for name and address matching;
> pay attention to country-specific address patterns; consider the
> precision-recall trade-off carefully — F_0.5 rewards precision more than
> recall; do not neglect singletons — correctly predicting "no match" is worth a
> full 1.0 on that entity; validate your own output format against the rules
> before submitting.

**Read:** external data is banned outright, with disqualification as the
penalty, which closes the open question from chunk 6 and makes the
model-versus-data boundary something to hold deliberately — pretrained weights
are permitted by chunk 6, but a downloaded gazetteer is external data even
though it resolves no business directly, and the asymmetry between a score hit
and disqualification argues for staying clearly inside the line. The tips are
informative rather than filler: they confirm the blocking read outright ("it
determines the upper bound of your recall"), confirm the singleton read, and
notably point at **classical string similarity — Jaccard, Levenshtein, TF-IDF
cosine — rather than embeddings**, despite chunk 6 permitting models up to 8B.
That suggests the organisers expect classical features to carry real weight.
One genuine tension: chunk 3 forbids hard-coding to {US, India} while chunk 9
advises attention to country-specific address patterns — reconcilable only if
country-conditional logic degrades gracefully to an unseen label.

---

## Environment — settled, not up for debate

Full detail in [README.md](README.md). Short version:

- Python 3.11 venv via `uv`. System Python is 3.14 — too new for the ML stack.
- `sagemaker` pinned `<3`; v3 removed `session`/`estimator`/`processing`.
- `sqlalchemy<2.1` mandatory — 2.1.0 ships a malformed `pyproject.toml`.
- Region `ap-south-1`, IAM user `bhuvi-dev` (not root).
- Public repo: `github.com/Bhuvilol/AmazonML`

**Open infra items:** GPU quota all zero, two requests pending
(`ml.g4dn.xlarge`, `ml.g5.2xlarge`, 2 each). Budget alarm not yet set.

---

## Decisions log

| # | Decision | Rationale |
|---|---|---|
| 1 | Region `ap-south-1`, not `ap-south-2` | 183 vs 141 services; Hyderabad lacks Textract, Rekognition, Comprehend, Amplify |
| 2 | SageMaker SDK pinned `<3` | v3 restructure; all reference material targets v2 |
| 3 | Training jobs over notebook instances | Self-terminating, per-second billing; notebooks bill while idle |
| 4 | Ask small on quota | Modest increases tend to auto-approve; large ones go to human review |
