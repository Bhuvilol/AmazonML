# EXPERIMENT 7 -- REVERSE (target -> S1) retrieval. RETRIEVAL ONLY, no model.
#
# METHODOLOGY NOTE, and the reason this is not the naive implementation:
# reverse retrieval's difficulty is set by the S1-side pool size. Our locked
# eval holds 15,000 S1 entities but production India holds 883,188. Scoring
# each target's top-k against only the eval 15k makes the task 59x (India) /
# 88x (US) easier than production and would produce a number that cannot
# transfer -- the same pool-size inflation that made our 120k-target probes
# read 0.976 against a true full-pool 0.9218.
#
# So BOTH parts below retrieve against the FULL S1 population:
#   A. RECOVERY (exact)  -- queries = the blocked-out true targets of eval
#      entities; does the correct eval S1 appear in that target's top-k drawn
#      from all 883k/1.32M S1 records?
#   B. COST (sampled)    -- queries = 200k random targets; how many NEW pairs
#      involving an eval S1 does reverse retrieval propose? Extrapolated to the
#      full target population.
# Exact numerator, sampled denominator, honest competition in both.
COMMIT = "b0fb516"
REPO = "https://github.com/Bhuvilol/AmazonML.git"
N_EVAL = 15000
N_COST_SAMPLE = 200000
KS = (1, 3, 5, 10)

import os, sys, json, subprocess, time
from pathlib import Path
import numpy as np

t0 = time.time()
def sh(c, check=True):
    print(f"$ {c}", flush=True); return subprocess.run(c, shell=True, check=check)
sh("nproc; free -g | head -2", check=False)
sh("pip install -q sparse_dot_topn rapidfuzz polars 2>&1 | tail -1", check=False)
CHECKOUT = Path("/tmp/AmazonML")
if not CHECKOUT.exists():
    sh(f"git clone {REPO} {CHECKOUT}"); sh(f"cd {CHECKOUT} && git checkout --quiet {COMMIT}")
SRC = CHECKOUT / "Lassi_submission/code/business_entity_resolution/src"
sys.path.insert(0, str(SRC))
act = subprocess.run(f"cd {CHECKOUT} && git rev-parse --short HEAD", shell=True,
                     capture_output=True, text=True).stdout.strip()
assert act.startswith(COMMIT); print(f"PINNED {act}", flush=True)

import polars as pl, scipy.sparse as sp
import blocking as B
from pipeline import build_partition_candidates, BlockingConfig, _READ_OPTS
from normalize import add_normalized_columns

def find(p):
    h = list(Path("/kaggle/input").rglob(p)); assert h, p; return h[0]
DATA = find("train_source1.tsv").parent
os.environ["LASSI_DATA_ROOT"] = str(DATA.parent)
OUT = Path("/kaggle/working")

gt = {}
with open(DATA / "train_ground_truth.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        sid, _, rest = line.partition("\t"); rest = rest.rstrip("\n")
        gt[sid] = set(rest.split(",")) if rest.strip() else set()

cfg = BlockingConfig()
report = {"pinned": act, "n_eval": N_EVAL, "n_cost_sample": N_COST_SAMPLE, "ks": list(KS)}

for COUNTRY in ("India", "US"):
    print("\n" + "=" * 74 + f"\n{COUNTRY}\n" + "=" * 74, flush=True)
    s1full = (pl.scan_csv(DATA / "train_source1.tsv", **_READ_OPTS)
                .filter(pl.col("country") == COUNTRY).collect())
    s1full = s1full.filter(pl.col("entity_id").is_in(list(gt)))
    ev = s1full.sample(n=N_EVAL, seed=777)
    ev_ids = ev["entity_id"].to_list()
    ev_set = set(ev_ids)
    s2 = (pl.scan_csv(DATA / "train_source2.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    s3 = (pl.scan_csv(DATA / "train_source3.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    n_s2 = s2.height
    pool = pl.concat([s2, s3]); del s2, s3
    print(f"  FULL S1 {s1full.height:,} | eval S1 {len(ev_ids):,} | targets {pool.height:,}", flush=True)

    s1n_full = add_normalized_columns(s1full.lazy()).collect()
    pooln = add_normalized_columns(pool.lazy()).collect(); del pool, s1full
    evn = add_normalized_columns(ev.lazy()).collect()
    tid = {e: i for i, e in enumerate(pooln["entity_id"].to_list())}
    s1_row_full = {e: i for i, e in enumerate(s1n_full["entity_id"].to_list())}
    ev_row = {e: i for i, e in enumerate(ev_ids)}
    NP = pooln.height

    # ---- current forward candidate set on the eval slice ----
    fwd, fstats = build_partition_candidates(evn, pooln, cfg)
    fkeys = np.sort(fwd.row.astype(np.int64) * np.int64(NP) + fwd.col)
    truth = [{tid[t] for t in gt[e] if t in tid} for e in ev_ids]
    tkeys = np.sort(np.array([r * np.int64(NP) + c for r, s in enumerate(truth) for c in s],
                             dtype=np.int64))
    fwd_hit = np.isin(tkeys, fkeys)
    tcard = np.array([len(gt[e]) for e in ev_ids])
    print(f"  forward recall {fwd_hit.mean():.4f}  cands {len(fkeys):,} "
          f"({len(fkeys)/N_EVAL:.1f}/S1)", flush=True)

    # ---- PART A: exact recovery, full S1 corpus ----
    blocked_mask = ~fwd_hit
    bt = tkeys[blocked_mask]
    b_rows = (bt // NP).astype(np.int64)          # eval S1 index
    b_cols = (bt % NP).astype(np.int64)           # target index
    print(f"  PART A: {len(bt):,} blocked true pairs; retrieving each target's "
          f"top-k over ALL {s1n_full.height:,} S1", flush=True)

    resA = {}
    for field_q, field_c, tag in (("name_block", "name_block", "name"),
                                  ("addr_block", "addr_block", "addr")):
        qt = [pooln[field_q][int(c)] for c in b_cols]
        src, tgt = B._vectorize(s1n_full[field_c].to_list(), qt,
                                max_df=cfg.max_df, ngram_range=cfg.ngram_range)
        m = B._topn_similarity(src, tgt, max(KS), 0.05, 20000, -1).tocsr()
        want = np.array([s1_row_full[ev_ids[int(r)]] for r in b_rows], dtype=np.int64)
        for k in KS:
            hit = 0
            for i in range(m.shape[0]):
                a, e = m.indptr[i], m.indptr[i + 1]
                if e > a:
                    dd = m.data[a:e]; ii = m.indices[a:e]
                    top = ii[np.argsort(-dd)[:k]]
                    if want[i] in top: hit += 1
            resA[f"{tag}_k{k}"] = {"recovered": hit, "of_blocked": int(len(bt)),
                                   "rate": hit / max(len(bt), 1)}
            print(f"    {tag} k={k:<3} recovered {hit:,}/{len(bt):,} = {hit/max(len(bt),1):.4f}", flush=True)
        del src, tgt, m

    # card-1 subset
    c1 = tcard[b_rows] == 1
    resA["blocked_card1"] = int(c1.sum())
    resA["blocked_s2"] = int((b_cols < n_s2).sum())
    resA["blocked_s3"] = int((b_cols >= n_s2).sum())

    # ---- PART B: candidate cost, sampled targets, full S1 corpus ----
    rng = np.random.default_rng(7)
    samp = rng.choice(NP, size=min(N_COST_SAMPLE, NP), replace=False)
    scale = NP / len(samp)
    print(f"  PART B: {len(samp):,} sampled targets vs ALL S1 (scale x{scale:.1f})", flush=True)
    resB = {}
    for field, tag in (("name_block", "name"), ("addr_block", "addr")):
        qt = [pooln[field][int(c)] for c in samp]
        src, tgt = B._vectorize(s1n_full[field].to_list(), qt,
                                max_df=cfg.max_df, ngram_range=cfg.ngram_range)
        m = B._topn_similarity(src, tgt, max(KS), 0.05, 20000, -1).tocsr()
        evrow_of_full = {s1_row_full[e]: ev_row[e] for e in ev_ids}
        for k in KS:
            pairs = set()
            for i in range(m.shape[0]):
                a, e = m.indptr[i], m.indptr[i + 1]
                if e <= a: continue
                dd = m.data[a:e]; ii = m.indices[a:e]
                for j in ii[np.argsort(-dd)[:k]]:
                    er = evrow_of_full.get(int(j))
                    if er is not None:
                        pairs.add(er * np.int64(NP) + int(samp[i]))
            est_total = len(pairs) * scale
            new = len(np.setdiff1d(np.array(sorted(pairs), dtype=np.int64), fkeys)) if pairs else 0
            resB[f"{tag}_k{k}"] = {
                "pairs_in_sample": len(pairs), "new_in_sample": new,
                "est_new_pairs_full": float(new * scale),
                "est_new_per_S1": float(new * scale / N_EVAL)}
            print(f"    {tag} k={k:<3} sample pairs {len(pairs):,} new {new:,} "
                  f"-> est {new*scale:,.0f} new pairs ({new*scale/N_EVAL:.1f}/S1)", flush=True)
        del src, tgt, m

    report[COUNTRY] = {"forward_recall": float(fwd_hit.mean()),
                       "forward_cands": int(len(fkeys)),
                       "forward_per_S1": float(len(fkeys) / N_EVAL),
                       "blocked_true_pairs": int(len(bt)),
                       "full_s1": int(s1n_full.height), "targets": NP,
                       "partA_recovery": resA, "partB_cost": resB}
    (OUT / "exp7.json").write_text(json.dumps(report, indent=2, default=float))
    del pooln, s1n_full, evn, fwd
    print(f"  [{COUNTRY} done {(time.time()-t0)/60:.1f} min]", flush=True)

(OUT / "exp7.json").write_text(json.dumps(report, indent=2, default=float))
print(f"\nDONE {(time.time()-t0)/60:.1f} min", flush=True)
