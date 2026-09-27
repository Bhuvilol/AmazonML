# Forensic audit v2 -- MEASURES ONLY, changes nothing.
# Pinned to b0fb516 (20-feature commit) so code matches the v2 model.
COMMIT = "b0fb516"
REPO = "https://github.com/Bhuvilol/AmazonML.git"
N_VALID = 15000

import os, sys, json, subprocess, time, csv
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
actual = subprocess.run(f"cd {CHECKOUT} && git rev-parse --short HEAD", shell=True,
                        capture_output=True, text=True).stdout.strip()
assert actual.startswith(COMMIT), f"PIN FAILED: {actual}"
print(f"PINNED {actual}", flush=True)

import polars as pl, lightgbm as lgb, scipy.sparse as sp
import blocking as B
from pipeline import build_partition_candidates, score_candidates, BlockingConfig, _READ_OPTS
from normalize import add_normalized_columns
from features import RecordArrays, compute_pair_features
import model as M, decide as DEC

def find(p):
    h = list(Path("/kaggle/input").rglob(p)); assert h, p; return h[0]
DATA = find("train_source1.tsv").parent
MODEL = find("model.txt")
THRESH = json.loads(find("threshold.json").read_text())["threshold"]
os.environ["LASSI_DATA_ROOT"] = str(DATA.parent)
OUT = Path("/kaggle/working")
booster = lgb.Booster(model_file=str(MODEL))
trained = M.TrainedModel(booster=booster, best_iteration=booster.num_trees())
NF = booster.num_feature()
print(f"threshold={THRESH}  model_features={NF}", flush=True)
assert NF == 20, f"expected the 20-feature v2 model, got {NF}"

gt = {}
with open(DATA / "train_ground_truth.tsv", encoding="utf-8") as f:
    f.readline()
    for line in f:
        sid, _, rest = line.partition("\t"); rest = rest.rstrip("\n")
        gt[sid] = set(rest.split(",")) if rest.strip() else set()

cfg = BlockingConfig()
report = {"threshold_v2": THRESH, "n_valid_per_country": N_VALID,
          "model_features": NF, "pinned_commit": actual,
          "blocking_config": {k: str(v) for k, v in cfg.__dict__.items()}}
fp_rows, fn_rows = [], []

def keyset(mat):
    """(row,col) pairs of a CSR matrix as a sorted int64 key array."""
    mat = mat.tocsr(); mat.sort_indices()
    r = np.repeat(np.arange(mat.shape[0], dtype=np.int64), np.diff(mat.indptr))
    return np.sort(r * np.int64(mat.shape[1]) + mat.indices)

for COUNTRY in ("India", "US"):
    print("\n" + "=" * 72 + f"\n{COUNTRY}\n" + "=" * 72, flush=True)
    s1 = (pl.scan_csv(DATA / "train_source1.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    s1 = s1.filter(pl.col("entity_id").is_in(list(gt))).sample(n=N_VALID, seed=777)
    s2 = (pl.scan_csv(DATA / "train_source2.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    s3 = (pl.scan_csv(DATA / "train_source3.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    n_s2 = s2.height
    pool = pl.concat([s2, s3]); del s2, s3
    s1n = add_normalized_columns(s1.lazy()).collect()
    pooln = add_normalized_columns(pool.lazy()).collect(); del pool
    NQ, NT = s1n.height, pooln.height
    print(f"  {NQ} queries x FULL pool {NT}", flush=True)

    tidx = {e: i for i, e in enumerate(pooln["entity_id"].to_list())}
    qid = s1n["entity_id"].to_list()
    truth = [{tidx[t] for t in gt[e] if t in tidx} for e in qid]
    tkeys = np.sort(np.array([r * np.int64(NT) + c
                              for r, s in enumerate(truth) for c in s], dtype=np.int64))
    is_s2 = (tkeys % NT) < n_s2
    ntrue, ntrue_s2, ntrue_s3 = len(tkeys), int(is_s2.sum()), int((~is_s2).sum())
    print(f"  true pairs {ntrue} (S2 {ntrue_s2} / S3 {ntrue_s3})", flush=True)

    # ---------- A. per-strategy attribution ----------
    sn, tn = s1n["name_block"].to_list(), pooln["name_block"].to_list()
    sa, ta = s1n["addr_block"].to_list(), pooln["addr_block"].to_list()
    snum, tnum = s1n["addr_nums"].to_list(), pooln["addr_nums"].to_list()

    strat = {}
    src, tgt = B._vectorize(tn, sn, max_df=cfg.max_df, ngram_range=cfg.ngram_range)
    strat["name_tfidf"] = B._topn_similarity(src, tgt, cfg.top_n_name, cfg.min_sim_name, 20000, -1)
    del src, tgt
    src, tgt = B._vectorize(ta, sa, max_df=cfg.max_df, ngram_range=cfg.ngram_range)
    strat["addr_tfidf"] = B._topn_similarity(src, tgt, cfg.top_n_addr, cfg.min_sim_addr, 20000, -1)
    del src, tgt
    strat["exact_name"] = B.exact_key_candidates(sn, tn, 100)
    strat["exact_addr"] = B.exact_key_candidates(sa, ta, 100)
    # I. numeric blocker -- measured, NOT enabled anywhere else
    src, tgt = B._vectorize(tnum, snum, min_df=1, max_df=cfg.max_df, analyzer="word")
    strat["numeric"] = B._topn_similarity(src, tgt, 20, 0.30, 20000, -1)
    del src, tgt

    ks = {k: keyset(v) for k, v in strat.items()}
    hit = {k: np.isin(tkeys, v, assume_unique=False) for k, v in ks.items()}
    A = {}
    for k in ks:
        A[k] = {"recall_overall": float(hit[k].mean()),
                "recall_s2": float(hit[k][is_s2].mean()),
                "recall_s3": float(hit[k][~is_s2].mean()),
                "candidates": int(len(ks[k]))}
    SHIPPED = ["name_tfidf", "addr_tfidf", "exact_name", "exact_addr"]
    union_shipped = np.zeros(ntrue, bool)
    for k in SHIPPED: union_shipped |= hit[k]
    union_all = union_shipped | hit["numeric"]
    A["_union_shipped"] = {"recall_overall": float(union_shipped.mean()),
                           "recall_s2": float(union_shipped[is_s2].mean()),
                           "recall_s3": float(union_shipped[~is_s2].mean())}
    A["_union_with_numeric"] = {"recall_overall": float(union_all.mean()),
                                "recall_s2": float(union_all[is_s2].mean()),
                                "recall_s3": float(union_all[~is_s2].mean())}
    # marginal: what each strategy uniquely adds over the union of the others
    for k in list(ks):
        others = np.zeros(ntrue, bool)
        for j in ks:
            if j != k and (j != "numeric"): others |= hit[j]
        A[k]["unique_vs_shipped_others"] = int((hit[k] & ~others).sum())
        A[k]["unique_pct_of_true"] = float((hit[k] & ~others).sum() / ntrue)
    # cumulative in order
    cum = np.zeros(ntrue, bool); A["_cumulative"] = {}
    for k in SHIPPED + ["numeric"]:
        cum |= hit[k]; A["_cumulative"][k] = float(cum.mean())
    # I. numeric marginal cost/benefit
    A["_numeric_marginal"] = {
        "extra_true_pairs": int((hit["numeric"] & ~union_shipped).sum()),
        "extra_true_pct": float((hit["numeric"] & ~union_shipped).sum() / ntrue),
        "extra_candidates": int(len(np.setdiff1d(ks["numeric"],
                                 np.unique(np.concatenate([ks[k] for k in SHIPPED]))))),
    }
    print("  A attribution:", json.dumps(
        {k: round(v["recall_overall"], 4) for k, v in A.items() if "recall_overall" in v}), flush=True)
    print("  I numeric marginal:", A["_numeric_marginal"], flush=True)
    del strat, ks

    # ---------- shipped candidate set ----------
    cands, stats = build_partition_candidates(s1n, pooln, cfg)
    sizes = np.diff(cands.indptr)
    ckeys = cands.row.astype(np.int64) * np.int64(NT) + cands.col
    blocked = np.isin(tkeys, np.sort(ckeys))
    rec = {"A_attribution": A,
           "recall_overall": float(blocked.mean()),
           "recall_s1_s2": float(blocked[is_s2].mean()),
           "recall_s1_s3": float(blocked[~is_s2].mean()),
           "true_pairs": ntrue, "true_s2": ntrue_s2, "true_s3": ntrue_s3,
           "candidates_total": int(len(cands)), "pool_size": NT, "n_s2": n_s2,
           "cand_mean": float(sizes.mean()), "cand_median": float(np.median(sizes)),
           "cand_p90": float(np.percentile(sizes, 90)), "cand_p95": float(np.percentile(sizes, 95)),
           "cand_p99": float(np.percentile(sizes, 99)), "cand_max": int(sizes.max()),
           "cand_zero_entities": int((sizes == 0).sum()),
           "reduction_ratio": float(1 - len(cands) / (NQ * NT))}

    # ---------- H. exact-match rates among candidates ----------
    sn_a = np.array(sn, dtype=object); tn_a = np.array(tn, dtype=object)
    sa_a = np.array(sa, dtype=object); ta_a = np.array(ta, dtype=object)
    en = np.fromiter((sn_a[r] == tn_a[c] and len(sn_a[r]) >= 4
                      for r, c in zip(cands.row, cands.col)), bool, len(cands))
    ea = np.fromiter((sa_a[r] == ta_a[c] and len(sa_a[r]) >= 4
                      for r, c in zip(cands.row, cands.col)), bool, len(cands))
    labels = np.isin(ckeys, tkeys).astype(np.int8)
    rec["H_exact_match"] = {
        "cand_exact_name": int(en.sum()), "match_rate_exact_name": float(labels[en].mean()) if en.any() else None,
        "cand_exact_addr": int(ea.sum()), "match_rate_exact_addr": float(labels[ea].mean()) if ea.any() else None,
        "cand_exact_both": int((en & ea).sum()),
        "match_rate_exact_both": float(labels[en & ea].mean()) if (en & ea).any() else None,
        "cand_exact_neither": int((~en & ~ea).sum()),
        "match_rate_exact_neither": float(labels[~en & ~ea].mean()),
        "base_match_rate": float(labels.mean())}
    print("  H exact-match:", json.dumps(rec["H_exact_match"]), flush=True)

    # ---------- score ----------
    p = score_candidates(s1n, pooln, cands, trained)

    # ---------- C. hard negatives ----------
    edges = [0.0, .2, .4, .6, .8, .9, 1.01]
    nb = np.digitize(cands.name_cos, edges) - 1
    ab = np.digitize(cands.addr_cos, edges) - 1
    neg = labels == 0
    rec["C_negatives"] = {
        "n_neg": int(neg.sum()), "n_pos": int(labels.sum()),
        "by_name_bin": {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(((nb == i) & neg).sum()) for i in range(6)},
        "by_addr_bin": {f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(((ab == i) & neg).sum()) for i in range(6)},
        "hard_neg_both_gt_0.6": int((neg & (cands.name_cos > .6) & (cands.addr_cos > .6)).sum()),
        "hard_neg_both_gt_0.8": int((neg & (cands.name_cos > .8) & (cands.addr_cos > .8)).sum()),
        "hard_neg_score_gt_thresh": int((neg & (p >= THRESH)).sum()),
        "pos_name_cos_mean": float(cands.name_cos[labels == 1].mean()),
        "neg_name_cos_mean": float(cands.name_cos[neg].mean()),
        "pos_addr_cos_mean": float(cands.addr_cos[labels == 1].mean()),
        "neg_addr_cos_mean": float(cands.addr_cos[neg].mean()),
        "pos_score_mean": float(p[labels == 1].mean()), "neg_score_mean": float(p[neg].mean())}
    print("  C negatives:", json.dumps(rec["C_negatives"])[:400], flush=True)

    # ---------- E. FN decomposition ----------
    pred = p >= THRESH
    tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
    fn_block = ntrue - int(labels.sum())
    fn_model = int(((~pred) & (labels == 1)).sum())
    bs2 = int((~blocked & is_s2).sum()); bs3 = int((~blocked & ~is_s2).sum())
    tks2 = np.isin(tkeys[is_s2], np.sort(ckeys[(p >= THRESH)]))
    tks3 = np.isin(tkeys[~is_s2], np.sort(ckeys[(p >= THRESH)]))
    prec = tp / max(tp + fp, 1); rall = tp / max(ntrue, 1)
    rec["E_fn_decomposition"] = {
        "true_pairs": ntrue, "tp": tp, "fp": fp,
        "fn_blocked_out": fn_block, "fn_blocked_out_pct": float(fn_block / ntrue),
        "fn_model_rejected": fn_model, "fn_model_rejected_pct": float(fn_model / ntrue),
        "fn_blocked_s2": bs2, "fn_blocked_s3": bs3,
        "emitted_recall_s2": float(tks2.mean()), "emitted_recall_s3": float(tks3.mean()),
        "pairwise_precision": prec, "pairwise_recall_vs_all_true": rall,
        "pairwise_f05": float(1.25 * prec * rall / (0.25 * prec + rall)) if prec + rall else 0.0}
    print("  E FN:", json.dumps(rec["E_fn_decomposition"]), flush=True)

    # ---------- D. thresholds, with an INDEPENDENT holdout ----------
    rng = np.random.default_rng(5)
    half = rng.permutation(NQ); selA, selB = set(half[:NQ//2].tolist()), set(half[NQ//2:].tolist())
    mA = np.fromiter((r in selA for r in range(NQ)), bool, NQ)
    def macro(th, mask=None, restrict=None):
        picks = DEC.select(cands.row, cands.col, p, NQ, threshold=float(th))
        idx = [i for i in range(NQ) if (mask is None or mask[i])]
        if restrict is None:
            return DEC.macro_f05_from_indices([picks[i] for i in idx], [truth[i] for i in idx])
        pk = [np.array([c for c in picks[i] if restrict(c)], dtype=np.int64) for i in idx]
        tr = [{t for t in truth[i] if restrict(t)} for i in idx]
        return DEC.macro_f05_from_indices(pk, tr)
    grid = [round(x, 3) for x in np.arange(0.10, 0.96, 0.05)]
    curve = [(t, macro(t)) for t in grid]
    cA = [(t, macro(t, mA)) for t in grid]
    bestA = max(cA, key=lambda kv: kv[1])[0]
    rec["D_threshold"] = {
        "current_threshold": THRESH,
        "curve_overall": curve,
        "curve_s2": [(t, macro(t, None, lambda c: c < n_s2)) for t in grid],
        "curve_s3": [(t, macro(t, None, lambda c: c >= n_s2)) for t in grid],
        "best_threshold_on_splitA": bestA,
        "macro_on_splitA_at_bestA": macro(bestA, mA),
        "macro_on_splitB_at_bestA": macro(bestA, ~mA),
        "macro_on_splitB_at_v2thresh": macro(THRESH, ~mA),
        "macro_overall_at_v2thresh": macro(THRESH)}
    print("  D threshold:", json.dumps({k: v for k, v in rec["D_threshold"].items()
                                        if not k.startswith("curve")}), flush=True)

    # ---------- F. singletons + G. exclusivity ----------
    picks_noexcl = DEC.select(cands.row, cands.col, p, NQ, threshold=float(THRESH),
                              enforce_exclusivity=False)
    picks = DEC.select(cands.row, cands.col, p, NQ, threshold=float(THRESH))
    assigned = np.concatenate([x for x in picks_noexcl if len(x)]) if any(len(x) for x in picks_noexcl) else np.array([], dtype=np.int64)
    uniq, cnts = np.unique(assigned, return_counts=True)
    n_before = int(sum(len(x) for x in picks_noexcl)); n_after = int(sum(len(x) for x in picks))
    rec["G_exclusivity"] = {
        "assignments_before": n_before, "assignments_after": n_after,
        "removed_by_resolver": n_before - n_after,
        "targets_claimed_by_multiple_s1": int((cnts > 1).sum()),
        "excess_claims": int((cnts - 1)[cnts > 1].sum())}
    pe = np.array([len(x) == 0 for x in picks]); ts = np.array([len(t) == 0 for t in truth])
    rec["F_singleton"] = {
        "true_singletons": int(ts.sum()), "true_singleton_rate": float(ts.mean()),
        "pred_singletons": int(pe.sum()), "pred_singleton_rate": float(pe.mean()),
        "singleton_precision": float((pe & ts).sum() / max(pe.sum(), 1)),
        "singleton_recall": float((pe & ts).sum() / max(ts.sum(), 1)),
        "false_merges_on_true_singletons": int(((~pe) & ts).sum()),
        "mean_emitted_nonempty": float(np.mean([len(x) for x in picks if len(x)])),
        "mean_true_nonsingleton": float(np.mean([len(t) for t in truth if t]))}
    print("  F singleton:", json.dumps(rec["F_singleton"]), flush=True)
    print("  G exclusivity:", json.dumps(rec["G_exclusivity"]), flush=True)

    # ---------- error examples ----------
    qn, qa2 = s1n["business_name"].to_list(), s1n["business_address"].to_list()
    pn, pa2 = pooln["business_name"].to_list(), pooln["business_address"].to_list()
    for idx, sink, kind in ((np.flatnonzero(pred & (labels == 0)), fp_rows, "FP"),
                            (np.flatnonzero((~pred) & (labels == 1)), fn_rows, "FN")):
        take = rng.choice(idx, size=min(150, len(idx)), replace=False) if len(idx) else []
        for i in take:
            r, c = int(cands.row[i]), int(cands.col[i])
            sink.append({"country": COUNTRY, "kind": kind, "score": round(float(p[i]), 5),
                         "name_cos": round(float(cands.name_cos[i]), 4),
                         "addr_cos": round(float(cands.addr_cos[i]), 4),
                         "exact_name": bool(en[i]), "exact_addr": bool(ea[i]),
                         "s1_name": qn[r], "s1_addr": qa2[r] or "",
                         "cand_name": pn[c], "cand_addr": pa2[c] or "",
                         "cand_source": "S2" if c < n_s2 else "S3",
                         "n_cands_entity": int(sizes[r]), "n_true_entity": len(truth[r])})

    report[COUNTRY] = rec
    (OUT / "audit.json").write_text(json.dumps(report, indent=2, default=float))
    del cands, p, labels, pooln, s1n
    print(f"  [{COUNTRY} done {(time.time()-t0)/60:.1f} min]", flush=True)

for name, rows in (("false_positives.tsv", fp_rows), ("false_negatives.tsv", fn_rows)):
    if rows:
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader(); w.writerows(rows)
(OUT / "audit.json").write_text(json.dumps(report, indent=2, default=float))
print(f"\nDONE {(time.time()-t0)/60:.1f} min", flush=True)
