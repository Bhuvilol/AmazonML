# POST-EXP2 RESIDUAL-FN DIAGNOSTIC -- measures only, changes nothing.
#
# Trains the ACCEPTED 43-feature model (20 base + 13 idf/exact + 10 competition
# local) on the same locked setup, then profiles what is still failing. No
# training, feature, blocking, threshold, or production change.
#
# ORIGINAL EXP2 HEADER FOLLOWS (setup is identical) ------------------------
# EXPERIMENT 2 -- competition-LOCAL features on top of the Experiment-1 model.
#
# Baseline is the 33-feature Exp1 model (20 base + 13 exact/IDF), NOT the old 20.
# Treatment adds 10 features describing a candidate relative to the others
# competing for the SAME S1 entity. tgt_degree and tgt_margin are deliberately
# EXCLUDED: their statistics are target-global, so their meaning changes between
# a 30k-entity training partition and a 660k-entity inference shard.
#
# PROVENANCE, asserted in code below: the 10 features are computed from
# (row, name_cos, addr_cos) only. They never receive `col`, labels, ground
# truth, split membership, model predictions, or any target-side array -- so
# they cannot encode target-global information even accidentally.
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
ALL_NAMES = list(FEATURE_NAMES) + NEW_NAMES          # 33 = Experiment 1 baseline

COMP_NAMES = ["rank_name", "rank_addr", "rank_comb",
              "marg_name", "marg_addr", "marg_comb",
              "rel_name", "rel_addr", "n_cands", "share_comb"]
ALL2_NAMES = ALL_NAMES + COMP_NAMES                  # 43 = Experiment 2 treatment


def competition_local(row, name_cos, addr_cos, n_entities):
    """S1-local competition features. Takes NO target-side input by construction.

    `col` is deliberately not a parameter: every statistic here is an aggregate
    over one entity's own candidate list, so the function is structurally
    incapable of encoding target-global counts. Identical at train and
    inference -- same function, same three inputs, and the per-entity candidate
    list is produced by the same unchanged blocking in both cases.
    """
    comb = name_cos + addr_cos
    best_n = np.zeros(n_entities, np.float32); np.maximum.at(best_n, row, name_cos)
    best_a = np.zeros(n_entities, np.float32); np.maximum.at(best_a, row, addr_cos)
    best_c = np.zeros(n_entities, np.float32); np.maximum.at(best_c, row, comb)
    count = np.bincount(row, minlength=n_entities).astype(np.float32)
    total = np.bincount(row, weights=comb, minlength=n_entities).astype(np.float32)

    def rank_within(vals):
        order = np.lexsort((-vals, row))
        ranks = np.empty(len(vals), np.float32)
        starts = np.searchsorted(row[order], np.arange(n_entities))
        ranks[order] = np.arange(len(vals)) - starts[row[order]]
        return ranks

    eps = np.float32(1e-6)
    out = np.empty((len(row), 10), dtype=np.float32)
    out[:, 0] = rank_within(name_cos)
    out[:, 1] = rank_within(addr_cos)
    out[:, 2] = rank_within(comb)
    out[:, 3] = name_cos - best_n[row]
    out[:, 4] = addr_cos - best_a[row]
    out[:, 5] = comb - best_c[row]
    out[:, 6] = name_cos / np.maximum(best_n[row], eps)
    out[:, 7] = addr_cos / np.maximum(best_a[row], eps)
    out[:, 8] = count[row]
    out[:, 9] = comb / np.maximum(total[row], eps)
    return out

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
          "exp1_features": ALL_NAMES, "competition_local": COMP_NAMES}
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
    full = np.hstack([base, new])                     # 33 -- Exp1 baseline

    # ---- competition-local 10, with provenance assertions ----
    import inspect as _inspect, ast as _ast
    _sig = list(_inspect.signature(competition_local).parameters)
    assert _sig == ["row", "name_cos", "addr_cos", "n_entities"], \
        f"provenance: unexpected signature {_sig}"
    assert "col" not in _sig, "provenance: competition_local must not receive col"
    # Identifier-level check, NOT substring matching: a naive substring scan
    # flags "minlength" for containing "gt". Walk the AST and collect every
    # Name/Attribute actually referenced, then assert none is a forbidden one.
    _fn = next(n for n in _ast.parse(_inspect.getsource(competition_local)).body
               if isinstance(n, _ast.FunctionDef))
    _names = {n.id for n in _ast.walk(_fn) if isinstance(n, _ast.Name)}
    _names |= {n.attr for n in _ast.walk(_fn) if isinstance(n, _ast.Attribute)}
    _forbidden = {"col", "label", "labels", "truth", "gt", "tkeys", "ckeys",
                  "predict", "inA", "m_tr", "m_ev", "pooln", "tgt_degree", "tgt_best"}
    _leak = _names & _forbidden
    assert not _leak, f"provenance: competition_local references {_leak}"
    print(f"  identifiers referenced: {sorted(_names - {'np'})}", flush=True)
    comp = competition_local(cands.row, cands.name_cos, cands.addr_cos, NQ)
    # share_comb must sum to 1 per entity -- proves it is an entity-local share.
    _chk = np.bincount(cands.row, weights=comp[:, 9].astype(np.float64), minlength=NQ)
    assert np.allclose(_chk[np.diff(cands.indptr) > 0], 1.0, atol=1e-3), \
        "provenance: share_comb does not sum to 1 per entity"
    print("  PROVENANCE OK: competition_local sees only (row, name_cos, addr_cos); "
          "share_comb sums to 1.0 per entity", flush=True)
    full2 = np.hstack([full, comp])                   # 43 -- Exp2 treatment
    print(f"  feature matrices: exp1 {full.shape}  exp2 {full2.shape}", flush=True)

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
    arm_pred, arm_score, arm_thr = {}, {}, {}
    for tag, X, names in (("exp2_43", full2, ALL2_NAMES),):
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
        arm_pred[tag] = pred                     # retained for cross-arm recovery
        arm_score[tag] = p                       # raw scores, for fixed-threshold compare
        arm_thr[tag] = bestA
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
        if False:
            rank_ev = comp[m_ev, 2]
            arms[tag]["rejected_by_rank"] = {
                str(k): int(((~pred) & (lab == 1) & (rank_ev == k)).sum()) for k in range(8)}
            arms[tag]["rejected_rank_ge8"] = int(((~pred) & (lab == 1) & (rank_ev >= 8)).sum())
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

    # ================= RESIDUAL DIAGNOSTIC =================
    lab_ev = labels[m_ev]
    p_ev = arm_score["exp2_43"]
    thr = arm_thr["exp2_43"]
    pred_ev = arm_pred["exp2_43"]
    rr = cands.row[m_ev]; cc = cands.col[m_ev]

    # feature columns on eval rows, by name
    colmap = {n: i for i, n in enumerate(ALL2_NAMES)}
    F = full2[m_ev]
    def col(n): return F[:, colmap[n]]

    PROFILE = ["name_cos", "addr_cos", "name_jaccard", "addr_jaccard",
               "name_idf_jaccard", "addr_idf_jaccard",
               "name_idf_coverage", "addr_idf_coverage",
               "rank_comb", "share_comb", "n_cands"]
    is_s3 = (cc >= n_s2)
    groups = {"A_accepted_true_positives": pred_ev & (lab_ev == 1),
              "B_remaining_model_rejected": (~pred_ev) & (lab_ev == 1),
              "C_false_positives": pred_ev & (lab_ev == 0)}
    prof = {}
    for gname, m in groups.items():
        e = {"n": int(m.sum())}
        if m.any():
            for f in PROFILE:
                v = col(f)[m]
                e[f] = [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])]
            e["share_S3"] = round(float(is_s3[m].mean()), 4)
            e["share_S2"] = round(1 - float(is_s3[m].mean()), 4)
            e["frac_name_cos_lt_0.01"] = round(float((col("name_cos")[m] < 0.01).mean()), 4)
            e["frac_addr_cos_lt_0.01"] = round(float((col("addr_cos")[m] < 0.01).mean()), 4)
        prof[gname] = e
    arms["profile"] = prof
    arms["country"] = COUNTRY
    print(f"  PROFILE sizes: " + ", ".join(f"{k}={v['n']:,}" for k, v in prof.items()), flush=True)

    # ---- residual FN: blocked-out vs candidate-but-below-threshold ----
    n_true_ev_tot = sum(len(truth[i]) for i in ev_rows)
    blocked_in = int((lab_ev == 1).sum())
    arms["residual_fn_split"] = {
        "true_pairs_eval": n_true_ev_tot,
        "blocked_out": n_true_ev_tot - blocked_in,
        "blocked_out_pct_of_all_true": (n_true_ev_tot - blocked_in) / n_true_ev_tot,
        "candidate_but_rejected": int(((~pred_ev) & (lab_ev == 1)).sum()),
        "candidate_but_rejected_pct_of_all_true":
            int(((~pred_ev) & (lab_ev == 1)).sum()) / n_true_ev_tot,
        "accepted": int((pred_ev & (lab_ev == 1)).sum()),
    }
    print(f"  RESIDUAL SPLIT: {json.dumps(arms['residual_fn_split'])}", flush=True)

    fnm = (~pred_ev) & (lab_ev == 1)
    rk = col("rank_comb"); sc = col("share_comb"); ai = col("addr_idf_jaccard")

    by_rank = {}
    for lo, hi, nm in ((0,1,"0"),(1,2,"1"),(2,3,"2"),(3,4,"3"),(4,6,"4-5"),
                       (6,8,"6-7"),(8,16,"8-15"),(16,10**9,"16+")):
        m = (rk >= lo) & (rk < hi)
        by_rank[nm] = {"rejected": int((fnm & m).sum()),
                       "all_true_in_bucket": int(((lab_ev == 1) & m).sum()),
                       "reject_rate": (float((fnm & m).sum() / max(((lab_ev==1)&m).sum(),1)))}
    arms["residual_by_rank"] = by_rank

    def quantile_buckets(vals, name):
        qs = np.percentile(vals[lab_ev == 1], [20, 40, 60, 80])
        out = {}
        edges = [-np.inf] + list(qs) + [np.inf]
        for i in range(5):
            m = (vals >= edges[i]) & (vals < edges[i+1])
            out[f"q{i+1}_[{edges[i]:.3f},{edges[i+1]:.3f})"] = {
                "rejected": int((fnm & m).sum()),
                "all_true": int(((lab_ev == 1) & m).sum()),
                "reject_rate": float((fnm & m).sum() / max(((lab_ev==1)&m).sum(), 1))}
        return out
    arms["residual_by_share_comb_q"] = quantile_buckets(sc, "share_comb")
    arms["residual_by_addr_idf_q"] = quantile_buckets(ai, "addr_idf_jaccard")
    arms["frac_residual_fn_name_cos_lt_0.01"] = float((col("name_cos")[fnm] < 0.01).mean()) if fnm.any() else None
    arms["frac_residual_fn_addr_cos_lt_0.01"] = float((col("addr_cos")[fnm] < 0.01).mean()) if fnm.any() else None
    print(f"  residual FN name_cos~0: {arms['frac_residual_fn_name_cos_lt_0.01']:.4f}"
          f"  addr_cos~0: {arms['frac_residual_fn_addr_cos_lt_0.01']:.4f}", flush=True)

    # ---- 200 highest-scoring residual FNs + what outranked them ----
    qn_r = s1n["business_name"].to_list(); qa_r = s1n["business_address"].to_list()
    pn_r = pooln["business_name"].to_list(); pa_r = pooln["business_address"].to_list()
    fn_idx = np.flatnonzero(fnm)
    if len(fn_idx):
        top = fn_idx[np.argsort(-p_ev[fn_idx])[:200]]
        # per-entity ordering of eval candidates by model score
        order = np.lexsort((-p_ev, rr))
        ent_start = {}
        for pos in range(len(order)):
            r = int(rr[order[pos]])
            if r not in ent_start: ent_start[r] = pos
        for i in top:
            r = int(rr[i]); c = int(cc[i])
            row = {"country": COUNTRY, "s1_id": qid[r], "s1_name": qn_r[r],
                   "s1_addr": qa_r[r] or "", "n_true": len(truth[r]),
                   "fn_cand_name": pn_r[c], "fn_cand_addr": pa_r[c] or "",
                   "fn_source": "S3" if c >= n_s2 else "S2",
                   "fn_score": round(float(p_ev[i]), 5), "threshold": thr}
            for f in PROFILE:
                row["fn_" + f] = round(float(col(f)[i]), 4)
            st = ent_start[r]
            k = 0
            while st + k < len(order) and int(rr[order[st + k]]) == r and k < 3:
                j = order[st + k]; cj = int(cc[j])
                row[f"top{k+1}_name"] = pn_r[cj]
                row[f"top{k+1}_score"] = round(float(p_ev[j]), 5)
                row[f"top{k+1}_is_true"] = int(lab_ev[j] == 1)
                k += 1
            diag_rows.append(row)
    print(f"  sampled {len(fn_idx[:200]) if len(fn_idx) else 0} residual FNs", flush=True)

    report[COUNTRY] = arms
    (OUT / "exp3.json").write_text(json.dumps(report, indent=2, default=float))
    del cands, base, new, full, full2, comp, labels, s1n, pooln, F
    print(f"  [{COUNTRY} done {(time.time()-t0)/60:.1f} min]", flush=True)

(OUT / "exp3.json").write_text(json.dumps(report, indent=2, default=float))

if diag_rows:
    keys = sorted({k for r in diag_rows for k in r})
    lead = ["country", "s1_id", "s1_name", "s1_addr", "n_true", "fn_cand_name",
            "fn_cand_addr", "fn_source", "fn_score", "threshold"]
    cols = lead + [k for k in keys if k not in lead]
    with open(OUT / "residual_fn_sample.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in diag_rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"wrote residual_fn_sample.tsv: {len(diag_rows)} rows", flush=True)
else:
    print("WARNING: no residual FN rows collected", flush=True)

print(f"\nDONE {(time.time()-t0)/60:.1f} min", flush=True)
