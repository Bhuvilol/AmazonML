# EXPERIMENT 4 -- cardinality-aware DECISION LAYER. No retraining, no new
# features, no blocking change. Same accepted 43-feature model.
#
# Motivation (Exp3): 80.5% India / 89.0% US of nearest-miss residual FNs were
# outranked ONLY by other TRUE candidates, and 0/400 by impostors alone. The
# resolver finds the right entity set and emits too few of its members.
#
# LABEL DISCIPLINE: every policy maps a DESCENDING-SORTED score vector to a
# prefix length. Policies receive scores only -- never labels, never true
# cardinality. True cardinality is used solely to BUCKET the reporting.
# Parameters are chosen on split A and evaluated unchanged on split B.
#
# ORIGINAL EXP3 HEADER FOLLOWS ---------------------------------------------
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

    # ================= EXPERIMENT 4: DECISION POLICIES =================
    lab_ev = labels[m_ev]
    p_ev = arm_score["exp2_43"]
    thr0 = arm_thr["exp2_43"]
    rr = cands.row[m_ev] - NT
    cc = cands.col[m_ev]
    NE = NQ - NT

    np.savez_compressed(OUT / f"scores_{COUNTRY}.npz", row=rr.astype(np.int32),
                        col=cc.astype(np.int32), p=p_ev.astype(np.float32),
                        label=lab_ev.astype(np.int8), thr0=np.float32(thr0))

    # Per-entity descending-sorted view. Built ONCE; every policy reads it.
    order = np.lexsort((-p_ev, rr))
    rs, ps, cs, ls = rr[order], p_ev[order], cc[order], lab_ev[order]
    bounds = np.searchsorted(rs, np.arange(NE + 1))
    ENT = []
    for i in range(NE):
        a, b = bounds[i], bounds[i + 1]
        ENT.append((ps[a:b], cs[a:b], ls[a:b]))
    true_card = np.array([len(truth[NT + i]) for i in range(NE)], dtype=np.int32)

    # ---------- policies: score vector -> prefix length. No labels. ----------
    def n_thresh(sv, T):
        return int(np.searchsorted(-sv, -T, side="right"))
    def n_topk_floor(sv, k, Tlo):
        return int(min(k, np.searchsorted(-sv, -Tlo, side="right")))
    def n_thresh_cap(sv, T, k):
        return int(min(k, np.searchsorted(-sv, -T, side="right")))
    def n_lead_ratio(sv, T, beta, Tlo):
        if len(sv) == 0 or sv[0] < T: return 0
        cut = max(beta * float(sv[0]), Tlo)
        return int(np.searchsorted(-sv, -cut, side="right"))
    def n_succ_ratio(sv, T, alpha, Tlo):
        if len(sv) == 0 or sv[0] < T: return 0
        n = 1
        while n < len(sv) and sv[n] >= alpha * sv[n - 1] and sv[n] >= Tlo:
            n += 1
        return n
    def n_gap(sv, T, delta, Tlo):
        if len(sv) == 0 or sv[0] < T: return 0
        n = 1
        while n < len(sv) and (sv[n - 1] - sv[n]) <= delta and sv[n] >= Tlo:
            n += 1
        return n

    def f05(pred_set, t):
        if not t: return 1.0 if not pred_set else 0.0
        if not pred_set: return 0.0
        tp = len(pred_set & t)
        if not tp: return 0.0
        P, R = tp / len(pred_set), tp / len(t)
        return 1.25 * P * R / (0.25 * P + R)

    truth_ev = [truth[NT + i] for i in range(NE)]
    rngp = np.random.default_rng(5)
    permE = rngp.permutation(NE)
    Amask = np.zeros(NE, bool); Amask[permE[:NE // 2]] = True

    def evaluate(fn, mask):
        tot = 0.0; cnt = 0
        for i in range(NE):
            if not mask[i]: continue
            sv, cv, _ = ENT[i]
            n = fn(sv)
            tot += f05(set(cv[:n].tolist()), truth_ev[i]); cnt += 1
        return tot / max(cnt, 1)

    POLICIES = []
    for T in (0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        POLICIES.append((f"thresh(T={T})", lambda sv, T=T: n_thresh(sv, T)))
    for k in (2, 3, 4, 5, 6, 8):
        for Tlo in (0.2, 0.35, 0.5):
            POLICIES.append((f"topk(k={k},Tlo={Tlo})",
                             lambda sv, k=k, Tlo=Tlo: n_topk_floor(sv, k, Tlo)))
    for T in (0.5, 0.6, 0.7):
        for k in (3, 4, 5, 6, 8):
            POLICIES.append((f"thresh_cap(T={T},k={k})",
                             lambda sv, T=T, k=k: n_thresh_cap(sv, T, k)))
    for T in (0.5, 0.6, 0.65, 0.7, 0.75):
        for beta in (0.5, 0.6, 0.7, 0.8, 0.9):
            for Tlo in (0.15, 0.25, 0.35):
                POLICIES.append((f"lead_ratio(T={T},b={beta},Tlo={Tlo})",
                                 lambda sv, T=T, b=beta, Tlo=Tlo: n_lead_ratio(sv, T, b, Tlo)))
    for T in (0.5, 0.6, 0.65, 0.7, 0.75):
        for al in (0.7, 0.8, 0.9, 0.95):
            for Tlo in (0.15, 0.25, 0.35):
                POLICIES.append((f"succ_ratio(T={T},a={al},Tlo={Tlo})",
                                 lambda sv, T=T, a=al, Tlo=Tlo: n_succ_ratio(sv, T, a, Tlo)))
    for T in (0.5, 0.6, 0.65, 0.7, 0.75):
        for dl in (0.05, 0.1, 0.2, 0.3):
            for Tlo in (0.15, 0.25, 0.35):
                POLICIES.append((f"gap(T={T},d={dl},Tlo={Tlo})",
                                 lambda sv, T=T, d=dl, Tlo=Tlo: n_gap(sv, T, d, Tlo)))
    print(f"  sweeping {len(POLICIES)} policies on split A ...", flush=True)

    resA = []
    for nm, fn in POLICIES:
        resA.append((nm, evaluate(fn, Amask), fn))
    resA.sort(key=lambda t: -t[1])
    fam = {}
    for nm, vA, fn in resA:
        f = nm.split("(")[0]
        if f not in fam: fam[f] = (nm, vA, fn)
    print("  best per family on A:", flush=True)
    for f, (nm, vA, _) in fam.items():
        print(f"    {f:<14}{nm:<40}A={vA:.4f}", flush=True)

    # LOCK on A, evaluate unchanged on B
    CUR = ("CURRENT thresh(T=%.2f)" % thr0, lambda sv: n_thresh(sv, thr0))
    finalists = [CUR] + [(nm, fn) for f, (nm, vA, fn) in fam.items()]
    out_pol = {}
    for nm, fn in finalists:
        vA = evaluate(fn, Amask); vB = evaluate(fn, ~Amask)
        # full-eval stats at this policy
        tp = fp = fn_ct = 0; pred_card = np.zeros(NE, np.int32)
        pe = np.zeros(NE, bool)
        per_bucket = {}
        for i in range(NE):
            sv, cv, lv = ENT[i]
            n = fn(sv)
            pred_card[i] = n
            pe[i] = (n == 0)
            sel = set(cv[:n].tolist()); t = truth_ev[i]
            tp += len(sel & t); fp += len(sel - t); fn_ct += len(t - sel)
        ts = true_card == 0
        BK = ((0,0,"0"),(1,1,"1"),(2,2,"2"),(3,3,"3"),(4,5,"4-5"),(6,7,"6-7"),(8,99,"8+"))
        for lo,hi,label in BK:
            m = (true_card >= lo) & (true_card <= hi) & (~Amask)
            if not m.any(): per_bucket[label] = None; continue
            sc = np.mean([f05(set(ENT[i][1][:fn(ENT[i][0])].tolist()), truth_ev[i])
                          for i in np.flatnonzero(m)])
            per_bucket[label] = {"n": int(m.sum()), "F05_B": round(float(sc), 4),
                                 "mean_pred_card": round(float(pred_card[m].mean()), 3),
                                 "mean_true_card": round(float(true_card[m].mean()), 3)}
        out_pol[nm] = {
            "F05_A": vA, "F05_B": vB, "TP": tp, "FP": fp, "FN": fn_ct,
            "pred_singletons": int(pe.sum()), "true_singletons": int(ts.sum()),
            "singleton_precision": round(float((pe & ts).sum() / max(pe.sum(), 1)), 4),
            "singleton_recall": round(float((pe & ts).sum() / max(ts.sum(), 1)), 4),
            "false_merges_on_true_singletons": int(((~pe) & ts).sum()),
            "mean_pred_card_nonempty": round(float(pred_card[pred_card > 0].mean()), 3),
            "by_true_cardinality_B": per_bucket,
        }
        print(f"  {nm:<42}A={vA:.4f}  B={vB:.4f}  TP={tp:,} FP={fp:,} "
              f"singlPrec={out_pol[nm]['singleton_precision']:.3f}", flush=True)
    arms["policies"] = out_pol
    arms["current_threshold"] = thr0

    report[COUNTRY] = arms
    (OUT / "exp4.json").write_text(json.dumps(report, indent=2, default=float))
    del cands, base, new, full, full2, comp, labels, s1n, pooln
    print(f"  [{COUNTRY} done {(time.time()-t0)/60:.1f} min]", flush=True)

(OUT / "exp4.json").write_text(json.dumps(report, indent=2, default=float))

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
