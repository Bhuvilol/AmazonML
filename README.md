# AmazonML — Hackathon

Solo entry. ~2-4 day deadline. AWS-based (sponsored credits).

**Status:** Phase 0 complete — environment and repo ready.
Awaiting problem statement before any modeling work begins.

---

## Setup

```bash
# one-time
brew install uv awscli
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# every session
source .venv/bin/activate
```

Verify the environment is sane:

```bash
python --version                       # must be 3.11.x, NOT 3.14
python -c "import numpy, pandas, sklearn, polars, boto3; print('ok')"
python -c "from sagemaker.estimator import Estimator; print('sagemaker ok')"
```

---

## Layout

| Path | Purpose | In git? |
|---|---|---|
| `data/raw/` | Original data, never edited | No — too large |
| `data/processed/` | Derived features, cached frames | No — regenerable |
| `notebooks/` | EDA and experiments | Yes |
| `src/` | Reusable code, importable from notebooks *and* SageMaker | Yes |
| `models/` | Checkpoints | No — large binaries |
| `submissions/` | Versioned outputs | Yes |

The `src/` vs `notebooks/` split matters: anything a SageMaker training job
needs must live in `src/`, because a remote job cannot import from a notebook.
Prototype in a notebook, then move the logic to `src/` as soon as it works.

---

## Environment gotchas

Three traps were hit and fixed during setup. See the comments in
`requirements.txt` for the full reasoning — the short version:

1. **Use Python 3.11, not the system 3.14.** The ML stack lags new CPython.
2. **SageMaker SDK is pinned to v2.** v3 removed `sagemaker.session`,
   `sagemaker.estimator` and `sagemaker.processing`. Every tutorial online
   targets v2. The v2 deprecation warning is expected and harmless.
3. **`sqlalchemy<2.1` is mandatory.** 2.1.0 has a malformed `pyproject.toml`
   that breaks the entire install.

---

## AWS

Console-first by preference: every step should be doable and visible in the
AWS Console. The CLI is installed as an accelerator for read-only checks, not
as the primary interface.

**Identity:** a dedicated IAM user with `AdministratorAccess` — *not* the root
account. Root cannot be constrained by any policy, so a leaked root key is
unrecoverable; an IAM key can be deleted and rotated.

### Region: `ap-south-1` (Mumbai) — decided, do not change casually

Quotas are **per-region**, so switching means re-requesting GPU quota and
waiting again. Chosen over the alternatives on measured service coverage
(from AWS's public regional services index, 2026-09-25):

| Region | Services | Notes |
|---|---|---|
| us-east-1 | 197 | Most coverage, cheapest — but slow uploads from India |
| us-west-2 | 194 | Near parity, often better GPU capacity |
| **ap-south-1** | **183** | **Chosen** — low latency from India, near-full coverage |
| ap-south-2 | 141 | Rejected — missing 42 services |

`ap-south-2` (Hyderabad) was rejected because it lacks **Textract,
Rekognition, Comprehend, Transcribe, Translate, Personalize, Kendra,
SageMaker Ground Truth, Amplify and App Runner**. The pre-trained AI services
are potential shortcuts to a strong baseline on an image or text task, and
Amplify/App Runner are the fastest routes to a demo judges can click. Mumbai
gives the same India latency with none of those gaps.

Before any training runs:

- [x] IAM user `bhuvi-dev` created; credentials configured; region `ap-south-1`
- [x] GPU quotas checked — **all zero**, as expected for this account
- [x] Quota increases filed (see below)
- [ ] Budget alarm set — SageMaker notebook instances bill while idle
- [ ] Kaggle fallback confirmed

### Quota status (filed 2026-09-25)

Every `ml.g*` / `ml.p*` quota and both EC2 GPU families read **0** — no GPU
could be launched at all. Requests filed:

| Quota | Code | Asked | Status |
|---|---|---|---|
| `ml.g4dn.xlarge` training job | `L-3F53BF0F` | 2 | PENDING |
| `ml.g5.2xlarge` training job | `L-2D6DEB3C` | 2 | PENDING |
| Studio `ml.g5.xlarge` notebook | `L-60470224` | 1 | blocked — see below |
| EC2 G and VT vCPUs | `L-DB2E81BA` | 8 | blocked — see below |

**AWS caps concurrent open quota requests; this account's cap is 2.** The
third and fourth requests failed with `QuotaExceededException`. File them once
a pending request resolves.

Two deliberate choices worth keeping:

- **Ask small.** Modest increases often auto-approve in minutes; large ones go
  to human review and can take days. Two instances is enough for solo work.
- **Prefer training jobs over notebook instances.** Training jobs terminate
  themselves and bill per second. Notebook instances bill while idle and are
  the most common way hackathon credits quietly disappear.

Check status:
```bash
aws service-quotas list-requested-service-quota-change-history \
  --status PENDING --output table \
  --query 'RequestedQuotas[].[QuotaName,DesiredValue,Status]'
```
Console: **Service Quotas → Quota request history**

The hardware exists in Mumbai — `g4dn`, `g5`, `g6` and `p4d.24xlarge` are all
offered. Quota is the only blocker.

### Cost reference

| Option | ~$/hr | Notes |
|---|---|---|
| SageMaker `ml.g4dn.xlarge` | 0.74 | T4 16GB — best value |
| SageMaker `ml.g5.xlarge` | 1.41 | A10G 24GB — faster |
| EC2 `g5.xlarge` | 1.01 | Cheaper, more setup |
| Kaggle | 0.00 | T4/P100, 30h/week — fallback |

Approximate and region-dependent. Confirm in-console before spending real
credit.

---

## Hardware note

Local machine is an Apple M1 with 8 GB RAM. That is a development box, not a
training box. Full-dataset training happens on AWS (or Kaggle as fallback).
Keep code portable between local and remote from the first commit.
