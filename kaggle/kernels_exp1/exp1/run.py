# EXPERIMENT 1 -- exact-match + IDF lexical features. Controlled A/B.
#
# Everything held constant except the feature matrix:
#   * same pinned code b0fb516 (20-feature baseline, no competition features)
#   * same candidate set (shipped BlockingConfig, untouched)
#   * same 15k/country locked eval entities (seed 777, identical to the audit)
#   * same A->B threshold protocol (rng seed 5, 50/50 entity split)
# Two arms are trained on the SAME training entities and scored on the SAME
# eval entities, so the only difference is 20 vs 33 columns.
COMMIT = "b0fb516"
REPO = "https://github.com/Bhuvilol/AmazonML.git"
N_EVAL, N_TRAIN = 15000, 15000

import os, sys, json, subprocess, time, math, csv, collections
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
assert actual.startswith(COMMIT), f"PIN FAILED {actual}"
print(f"PINNED {actual}", flush=True)

import polars as pl, lightgbm as lgb
from pipeline import build_partition_candidates, BlockingConfig, _READ_OPTS
from normalize import add_normalized_columns
from features import RecordArrays, compute_pair_features, FEATURE_NAMES
import decide as DEC

assert len(FEATURE_NAMES) == 20, f"expected 20 baseline features, got {len(FEATURE_NAMES)}"

NEW_NAMES = [
    "exact_name_norm", "exact_addr_norm", "exact_both_norm",
    "exact_name_block", "exact_addr_block",
    "name_jaccard", "addr_jaccard",
    "name_idf_jaccard", "addr_idf_jaccard",
    "name_idf_coverage", "addr_idf_coverage",
    "name_max_shared_idf", "addr_max_shared_idf",
]
ALL_NAMES = list(FEATURE_NAMES) + NEW_NAMES

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
report = {"pinned": actual, "n_eval": N_EVAL, "n_train": N_TRAIN,
          "baseline_features": list(FEATURE_NAMES), "new_features": NEW_NAMES}
diag_rows = []

def idf_table(texts):
    """Word-level IDF over the target corpus -- same corpus blocking fits on."""
    df = collections.Counter()
    for t in texts:
        if t:
            df.update(set(t.split()))
    n = len(texts)
    return {tok: math.log(n / (1 + c)) + 1.0 for tok, c in df.items()}

for COUNTRY in ("India", "US"):
    print("\n" + "=" * 72 + f"\n{COUNTRY}\n" + "=" * 72, flush=True)
    s1 = (pl.scan_csv(DATA / "train_source1.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    s1 = s1.filter(pl.col("entity_id").is_in(list(gt)))
    ev = s1.sample(n=N_EVAL, seed=777)                       # identical to the audit
    rest = s1.filter(~pl.col("entity_id").is_in(ev["entity_id"].implode()))
    tr = rest.sample(n=min(N_TRAIN, rest.height), seed=1234) # disjoint from eval
    s1all = pl.concat([tr, ev])                              # rows 0..NT-1 train, NT.. eval
    NT = tr.height
    print(f"  train {NT}  eval {ev.height}  (disjoint)", flush=True)

    s2 = (pl.scan_csv(DATA / "train_source2.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    s3 = (pl.scan_csv(DATA / "train_source3.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    n_s2 = s2.height
    pool = pl.concat([s2, s3]); del s2, s3
    s1n = add_normalized_columns(s1all.lazy()).collect()
    pooln = add_normalized_columns(pool.lazy()).collect(); del pool, s1all, s1, rest
    NQ, NP = s1n.height, pooln.height
    print(f"  {NQ} queries x full pool {NP}", flush=True)

    cands, stats = build_partition_candidates(s1n, pooln, cfg)
    print(f"  blocking: {stats.describe()}", flush=True)

    tidx = {e: i for i, e in enumerate(pooln["entity_id"].to_list())}
    qid = s1n["entity_id"].to_list()
    truth = [{tidx[t] for t in gt[e] if t in tidx} for e in qid]
    tkeys = np.sort(np.array([r * np.int64(NP) + c for r, s in enumerate(truth) for c in s],
                             dtype=np.int64))
    ckeys = cands.row.astype(np.int64) * np.int64(NP) + cands.col
    labels = np.isin(ckeys, tkeys).astype(np.int8)
    print(f"  candidates {len(cands):,}  positives {labels.sum():,}", flush=True)

    # ---- baseline 20 ----
    left, right = RecordArrays.from_frame(s1n), RecordArrays.from_frame(pooln)
    base = compute_pair_features(left, right, cands.row, cands.col,
                                 cands.name_cos, cands.addr_cos)
    assert base.shape[1] == 20

    # ---- new 13 ----
    qn = s1n["name_norm"].to_list();  qa = s1n["addr_norm"].to_list()
    pn = pooln["name_norm"].to_list(); pa = pooln["addr_norm"].to_list()
    qnb = s1n["name_block"].to_list(); qab = s1n["addr_block"].to_list()
    pnb = pooln["name_block"].to_list(); pab = pooln["addr_block"].to_list()
    print("  building IDF over target corpus ...", flush=True)
    idf_n, idf_a = idf_table(pn), idf_table(pa)

    used = np.unique(cands.col)
    qset_n = [set(x.split()) if x else set() for x in qn]
    qset_a = [set(x.split()) if x else set() for x in qa]
    tset_n = {}; tset_a = {}
    for c in used.tolist():
        tset_n[c] = set(pn[c].split()) if pn[c] else set()
        tset_a[c] = set(pa[c].split()) if pa[c] else set()
    print(f"  token sets built for {len(used):,} referenced targets", flush=True)

    M = len(cands)
    new = np.zeros((M, 13), dtype=np.float32)
    for i in range(M):
        r = int(cands.row[i]); c = int(cands.col[i])
        a_n, b_n = qset_n[r], tset_n[c]
        a_a, b_a = qset_a[r], tset_a[c]
        en = 1.0 if (qn[r] and qn[r] == pn[c]) else 0.0
        ea = 1.0 if (qa[r] and qa[r] == pa[c]) else 0.0
        new[i, 0] = en
        new[i, 1] = ea
        new[i, 2] = 1.0 if (en and ea) else 0.0
        new[i, 3] = 1.0 if (qnb[r] and qnb[r] == pnb[c]) else 0.0
        new[i, 4] = 1.0 if (qab[r] and qab[r] == pab[c]) else 0.0
        sh_n = a_n & b_n; un_n = a_n | b_n
        sh_a = a_a & b_a; un_a = a_a | b_a
        new[i, 5] = len(sh_n) / len(un_n) if un_n else 0.0
        new[i, 6] = len(sh_a) / len(un_a) if un_a else 0.0
        if un_n:
            ws = sum(idf_n.get(t, 1.0) for t in sh_n)
            new[i, 7] = ws / sum(idf_n.get(t, 1.0) for t in un_n)
            qw = sum(idf_n.get(t, 1.0) for t in a_n)
            new[i, 9] = ws / qw if qw else 0.0
            new[i, 11] = max((idf_n.get(t, 1.0) for t in sh_n), default=0.0)
        if un_a:
            ws = sum(idf_a.get(t, 1.0) for t in sh_a)
            new[i, 8] = ws / sum(idf_a.get(t, 1.0) for t in un_a)
            qw = sum(idf_a.get(t, 1.0) for t in a_a)
            new[i, 10] = ws / qw if qw else 0.0
            new[i, 12] = max((idf_a.get(t, 1.0) for t in sh_a), default=0.0)
        if i and i % 400000 == 0:
            print(f"    new-features {i:,}/{M:,}  ({(time.time()-t0)/60:.0f} min)", flush=True)
    full = np.hstack([base, new])
    print(f"  feature matrices: base {base.shape}  full {full.shape}", flush=True)

    # ---- train both arms on the SAME training rows ----
    m_tr = cands.row < NT
    m_ev = ~m_tr
    ev_rows = np.arange(NT, NQ)
    P = {"objective": "binary", "metric": ["binary_logloss", "average_precision"],
         "learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200,
         "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
         "lambda_l2": 1.0, "num_threads": 0, "verbosity": -1, "seed": 42,
         "deterministic": True}

    rng = np.random.default_rng(5)                 # identical protocol to the audit
    perm = rng.permutation(len(ev_rows))
    inA = np.zeros(NQ, bool); inA[ev_rows[perm[:len(ev_rows)//2]]] = True

    arms = {}
    for tag, X, names in (("baseline20", base, list(FEATURE_NAMES)),
                          ("plus13", full, ALL_NAMES)):
        b = lgb.train(P, lgb.Dataset(X[m_tr], label=labels[m_tr], feature_name=names),
                      num_boost_round=600)
        p = b.predict(X[m_ev]).astype(np.float32)
        rr, cc = cands.row[m_ev], cands.col[m_ev]
        lab = labels[m_ev]

        def macro(th, mask):
            picks = DEC.select(rr - NT, cc, p, NQ - NT, threshold=float(th))
            idx = [i for i in range(NQ - NT) if mask[i + NT]]
            return DEC.macro_f05_from_indices([picks[i] for i in idx],
                                              [truth[i + NT] for i in idx])
        grid = [round(x, 3) for x in np.arange(0.10, 0.96, 0.05)]
        cA = [(t, macro(t, inA)) for t in grid]
        bestA = max(cA, key=lambda kv: kv[1])[0]
        fA = macro(bestA, inA); fB = macro(bestA, ~inA)
        pred = p >= bestA
        n_true_ev = sum(len(truth[i]) for i in ev_rows)
        fn_model = int(((~pred) & (lab == 1)).sum())
        arms[tag] = {
            "best_threshold_on_A": bestA, "F05_A": fA, "F05_B": fB,
            "FP": int((pred & (lab == 0)).sum()),
            "TP": int((pred & (lab == 1)).sum()),
            "fn_model_rejected": fn_model,
            "fn_model_rejected_pct_of_all_true": fn_model / n_true_ev,
            "true_pairs_eval": n_true_ev,
            "importance": sorted(zip(names, b.feature_importance("gain").tolist()),
                                 key=lambda kv: -kv[1])[:20],
        }
        print(f"  {tag}: A={fA:.4f} B={fB:.4f} thr={bestA} FP={arms[tag]['FP']} "
              f"FN_model={fn_model} ({100*arms[tag]['fn_model_rejected_pct_of_all_true']:.2f}%)",
              flush=True)

        # ---- FN breakdown by exact-match class + diagnostic distributions ----
        if tag == "baseline20":
            enm = new[m_ev, 0].astype(bool); eam = new[m_ev, 1].astype(bool)
            fnm = (~pred) & (lab == 1); tpm = pred & (lab == 1); fpm = pred & (lab == 0)
            arms[tag]["fn_by_exact"] = {
                "exact_name_only": int((fnm & enm & ~eam).sum()),
                "exact_addr_only": int((fnm & eam & ~enm).sum()),
                "exact_both": int((fnm & enm & eam).sum()),
                "neither": int((fnm & ~enm & ~eam).sum())}
            nc, ac = cands.name_cos[m_ev], cands.addr_cos[m_ev]
            nj, aj = new[m_ev, 5], new[m_ev, 6]
            order = np.lexsort((-p, rr)); rank = np.empty(len(p), np.float32)
            starts = np.searchsorted(rr[order], np.arange(NT, NQ))
            rank[order] = np.arange(len(p)) - starts[rr[order] - NT]
            ncand = np.diff(cands.indptr)[rr]
            def dist(m, nm):
                if not m.any(): return {"n": 0}
                q = lambda v: [round(float(x), 4) for x in np.percentile(v[m], [10, 50, 90])]
                return {"n": int(m.sum()), "name_cos_p10_50_90": q(nc), "addr_cos_p10_50_90": q(ac),
                        "name_jaccard_p10_50_90": q(nj), "addr_jaccard_p10_50_90": q(aj),
                        "rank_p10_50_90": q(rank), "n_cands_p10_50_90": q(ncand),
                        "exact_name_rate": round(float(enm[m].mean()), 4),
                        "exact_addr_rate": round(float(eam[m].mean()), 4)}
            arms[tag]["diagnostic"] = {"model_rejected_positives": dist(fnm, "FN"),
                                       "accepted_true_positives": dist(tpm, "TP"),
                                       "false_positives": dist(fpm, "FP")}
            print(f"  FN by exact class: {arms[tag]['fn_by_exact']}", flush=True)

    a, bb = arms["baseline20"], arms["plus13"]
    arms["DELTA"] = {"F05_A": bb["F05_A"] - a["F05_A"], "F05_B": bb["F05_B"] - a["F05_B"],
                     "FP": bb["FP"] - a["FP"],
                     "fn_model_pct": bb["fn_model_rejected_pct_of_all_true"]
                                     - a["fn_model_rejected_pct_of_all_true"]}
    print(f"  DELTA A={arms['DELTA']['F05_A']:+.4f}  B={arms['DELTA']['F05_B']:+.4f}", flush=True)
    report[COUNTRY] = arms
    (OUT / "exp1.json").write_text(json.dumps(report, indent=2, default=float))
    del cands, base, new, full, labels, s1n, pooln
    print(f"  [{COUNTRY} done {(time.time()-t0)/60:.1f} min]", flush=True)

(OUT / "exp1.json").write_text(json.dumps(report, indent=2, default=float))
print(f"\nDONE {(time.time()-t0)/60:.1f} min", flush=True)
