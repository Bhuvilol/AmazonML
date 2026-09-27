# EXPERIMENT 6 (PHASE 1) -- learned normalisation, RETRIEVAL ONLY.
#
# No model retraining, no new resolver features. Measures whether a training-
# pair-derived normalisation view retrieves true pairs the five existing views
# cannot, and at what candidate cost.
#
# LEAKAGE CONTROL
#   * mappings are mined ONLY from training S1 entities (eval 15k seed-777
#     excluded by construction);
#   * mutual exclusivity is perfect in this dataset (0 of 7.6M targets claimed
#     by >1 S1), so excluding an eval entity also excludes its true targets
#     from mining -- the barrier is structural, not merely procedural;
#   * eval target strings enter only as inference corpus for TF-IDF, never as
#     mapping evidence;
#   * no external data of any kind.
COMMIT = "b0fb516"
REPO = "https://github.com/Bhuvilol/AmazonML.git"
N_EVAL = 15000
N_MINE = 250000          # training entities used for mapping estimation
MIN_SUPPORT = 10
MIN_CONF = 0.6

import os, sys, json, subprocess, time, collections
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
report = {"pinned": act, "n_eval": N_EVAL, "n_mine": N_MINE,
          "min_support": MIN_SUPPORT, "min_conf": MIN_CONF}

def _plausible(y, x):
    """Guard against spurious alignments that co-occur without being variants.

    Two admissible relations, both observed in the mined data:
      * a corruption/typo/transliteration of the same token -- high edit
        similarity ('privae'->'private', 'mhaaraassttr'->'maharashtra');
      * a short abbreviation sharing the initial ('mh'->'maharashtra').
    Everything else is rejected. Without this, support+confidence alone
    admitted 'llp'->'group' and 'calcutta'->'no'.
    """
    from rapidfuzz.distance import Indel
    if len(y) <= 3 and y[0] == x[0]:
        return True
    return Indel.normalized_similarity(y, x) >= 0.55


def mine(pairs):
    """Mine TARGET-token -> S1-token rewrites from true training pairs.

    Direction matters: Source 1 is the deduplicated clean reference and the
    targets carry the corruption, so targets are rewritten toward S1. The
    reverse direction is one-to-many (one clean token has many corrupted
    forms) and picking a single image of it is wrong.

    Only 1-to-1 alignments count -- pairs where each side has exactly one
    unmatched token -- so the correspondence is unambiguous.

    NO transitive closure. An earlier union-find version chained
    south/west/east/central -> 'dillii', which would have made "South Delhi"
    and "West Delhi" identical strings.
    """
    co = collections.Counter(); seen = collections.Counter()
    for a, b in pairs:
        if not a or not b: continue
        ta, tb = set(a.split()), set(b.split())
        da, db = ta - tb, tb - ta
        if len(da) == 1 and len(db) == 1:
            x, y = da.pop(), db.pop()
            if x == y or len(x) < 2 or len(y) < 2: continue
            co[(y, x)] += 1; seen[y] += 1
    keep = {}
    for (y, x), c in co.items():
        if c >= MIN_SUPPORT and c / seen[y] >= MIN_CONF and _plausible(y, x):
            keep[y] = x
    keep = {y: x for y, x in keep.items() if x not in keep}   # no 2-hop chains
    return keep, len(co), len(keep)


for COUNTRY in ("India", "US"):
    print("\n" + "=" * 74 + f"\n{COUNTRY}\n" + "=" * 74, flush=True)
    s1raw = (pl.scan_csv(DATA / "train_source1.tsv", **_READ_OPTS)
               .filter(pl.col("country") == COUNTRY).collect())
    s1raw = s1raw.filter(pl.col("entity_id").is_in(list(gt)))
    ev_ids = set(s1raw.sample(n=N_EVAL, seed=777)["entity_id"].to_list())
    mine_pool = s1raw.filter(~pl.col("entity_id").is_in(list(ev_ids)))
    mine_src = mine_pool.sample(n=min(N_MINE, mine_pool.height), seed=99)
    assert not (set(mine_src["entity_id"].to_list()) & ev_ids), "LEAK: eval entity in mining set"
    print(f"  mining from {mine_src.height:,} training entities (eval {len(ev_ids):,} excluded)", flush=True)

    s2 = (pl.scan_csv(DATA / "train_source2.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    s3 = (pl.scan_csv(DATA / "train_source3.tsv", **_READ_OPTS)
            .filter(pl.col("country") == COUNTRY).collect())
    n_s2 = s2.height
    poolraw = pl.concat([s2, s3]); del s2, s3
    # non-Latin flag from the RAW string, before transliteration
    nonlatin_tgt = np.array([any(ord(ch) > 127 for ch in (x or ""))
                             for x in poolraw["business_name"].to_list()])

    s1n = add_normalized_columns(s1raw.lazy()).collect()
    pooln = add_normalized_columns(poolraw.lazy()).collect(); del poolraw
    tid = {e: i for i, e in enumerate(pooln["entity_id"].to_list())}
    sid_row = {e: i for i, e in enumerate(s1n["entity_id"].to_list())}

    nm_all = s1n["name_norm"].to_list(); ad_all = s1n["addr_norm"].to_list()
    pnm = pooln["name_norm"].to_list();  pad = pooln["addr_norm"].to_list()
    np_pairs, ap_pairs = [], []
    for e in mine_src["entity_id"].to_list():
        r = sid_row[e]
        for t in gt[e]:
            j = tid.get(t)
            if j is None: continue
            np_pairs.append((nm_all[r], pnm[j])); ap_pairs.append((ad_all[r], pad[j]))
    canon_n, cand_n, kept_n = mine(np_pairs)
    canon_a, cand_a, kept_a = mine(ap_pairs)
    print(f"  name mappings : {kept_n:,} kept of {cand_n:,} observed alignments", flush=True)
    print(f"  addr mappings : {kept_a:,} kept of {cand_a:,} observed alignments", flush=True)
    ex = list(canon_n.items())[:12]
    print(f"  sample name mappings: {ex}", flush=True)
    report.setdefault(COUNTRY, {})["mappings"] = {
        "name_kept": kept_n, "name_observed": cand_n,
        "addr_kept": kept_a, "addr_observed": cand_a,
        "name_sample": [list(x) for x in ex],
        "addr_sample": [list(x) for x in list(canon_a.items())[:12]]}

    def rewrite(texts, canon):
        if not canon: return list(texts)
        g = canon.get
        return [" ".join(sorted(g(t, t) for t in (x.split() if x else []))) for x in texts]

    # eval slice only
    ev_mask = np.array([e in ev_ids for e in s1n["entity_id"].to_list()])
    s1e = s1n.filter(pl.Series(ev_mask))
    qid = s1e["entity_id"].to_list()
    truth = [{tid[t] for t in gt[e] if t in tid} for e in qid]
    tkeys = np.sort(np.array([r * np.int64(pooln.height) + c
                              for r, s in enumerate(truth) for c in s], dtype=np.int64))
    tcard = np.array([len(gt[e]) for e in qid])
    NEv, NP = s1e.height, pooln.height
    print(f"  eval {NEv} x pool {NP}; true pairs {len(tkeys):,}", flush=True)

    base_c, base_stats = build_partition_candidates(s1e, pooln, cfg)
    bkeys = np.sort(base_c.row.astype(np.int64) * np.int64(NP) + base_c.col)
    base_hit = np.isin(tkeys, bkeys)
    print(f"  baseline recall {base_hit.mean():.4f}  cands {len(base_c):,}", flush=True)

    def view(qtexts, ttexts, topn, minsim):
        src, tgt = B._vectorize(ttexts, qtexts, max_df=cfg.max_df, ngram_range=cfg.ngram_range)
        m = B._topn_similarity(src, tgt, topn, minsim, 20000, -1)
        m = m.tocsr(); m.sort_indices()
        r = np.repeat(np.arange(m.shape[0], dtype=np.int64), np.diff(m.indptr))
        return np.sort(r * np.int64(NP) + m.indices)
    nk = view(rewrite(s1e["name_block"].to_list(), canon_n),
              rewrite(pooln["name_block"].to_list(), canon_n), cfg.top_n_name, cfg.min_sim_name)
    ak = view(rewrite(s1e["addr_block"].to_list(), canon_a),
              rewrite(pooln["addr_block"].to_list(), canon_a), cfg.top_n_addr, cfg.min_sim_addr)
    newk = np.union1d(nk, ak)
    allk = np.union1d(bkeys, newk)
    new_hit = np.isin(tkeys, allk)
    extra_c = len(allk) - len(bkeys)
    extra_t = int(new_hit.sum() - base_hit.sum())

    tgt_of = (tkeys % NP).astype(np.int64)
    is_s2 = tgt_of < n_s2
    nl = nonlatin_tgt[tgt_of]
    c1 = np.isin((tkeys // NP).astype(np.int64), np.flatnonzero(tcard == 1))
    prev_blocked = ~base_hit
    rec = prev_blocked & new_hit
    sizes_new = np.bincount((allk // NP).astype(np.int64), minlength=NEv)
    e = {
      "baseline_recall": float(base_hit.mean()), "new_recall": float(new_hit.mean()),
      "baseline_recall_card1": float(base_hit[c1].mean()), "new_recall_card1": float(new_hit[c1].mean()),
      "baseline_cands": int(len(bkeys)), "new_cands": int(len(allk)),
      "baseline_mean_per_s1": float(len(bkeys)/NEv), "new_mean_per_s1": float(len(allk)/NEv),
      "new_p50": float(np.percentile(sizes_new,50)), "new_p95": float(np.percentile(sizes_new,95)),
      "new_p99": float(np.percentile(sizes_new,99)), "new_max": int(sizes_new.max()),
      "extra_candidates": extra_c, "extra_true_pairs": extra_t,
      "cost_per_true_pair": (extra_c/extra_t) if extra_t else None,
      "card1_prev_blocked": int((prev_blocked & c1).sum()),
      "card1_recovered": int((rec & c1).sum()),
      "card1_recovery_pct": float((rec&c1).sum()/max((prev_blocked&c1).sum(),1)),
      "recovery_by_source": {
         "S2": {"prev_blocked": int((prev_blocked&is_s2).sum()), "recovered": int((rec&is_s2).sum())},
         "S3": {"prev_blocked": int((prev_blocked&~is_s2).sum()), "recovered": int((rec&~is_s2).sum())}},
      "recovery_by_script": {
         "non_latin_target": {"prev_blocked": int((prev_blocked&nl).sum()), "recovered": int((rec&nl).sum())},
         "latin_target": {"prev_blocked": int((prev_blocked&~nl).sum()), "recovered": int((rec&~nl).sum())}},
    }
    report[COUNTRY].update(e)
    print("  " + json.dumps({k: e[k] for k in
        ("baseline_recall","new_recall","baseline_recall_card1","new_recall_card1",
         "extra_candidates","extra_true_pairs","cost_per_true_pair",
         "card1_prev_blocked","card1_recovered")}), flush=True)
    (OUT / "exp6.json").write_text(json.dumps(report, indent=2, default=float))
    del base_c, pooln, s1n, s1e
    print(f"  [{COUNTRY} done {(time.time()-t0)/60:.1f} min]", flush=True)

(OUT / "exp6.json").write_text(json.dumps(report, indent=2, default=float))
print(f"\nDONE {(time.time()-t0)/60:.1f} min", flush=True)
