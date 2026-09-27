"""Production distribution sanity check, per country, France broken out.

Looks for pipeline failure rather than for France resembling India/US:
zero candidates, candidate explosion, everything-singleton, absurd match
rates, invalid ids, duplicate rows.
"""
import sys, collections, polars as pl
sys.path.insert(0, "Lassi_submission/code/business_entity_resolution/src")
from pipeline import _READ_OPTS

OUT = sys.argv[1] if len(sys.argv) > 1 else "output_final"
D = "student_resource/dataset/test"

country = {}
for s in (1,):
    f = pl.scan_csv(f"{D}/test_source{s}.tsv", **_READ_OPTS).select(["entity_id","country"]).collect()
    country = dict(zip(f["entity_id"].to_list(), f["country"].to_list()))
valid = set()
for s in (2, 3):
    valid |= set(pl.scan_csv(f"{D}/test_source{s}.tsv", **_READ_OPTS)
                   .select("entity_id").collect()["entity_id"].to_list())
print(f"valid S2/S3 ids: {len(valid):,}\n")

def scan(path, col):
    st = collections.defaultdict(lambda: {"rows":0,"empty":0,"ids":0,"s2":0,"s3":0,
                                          "inv":0,"dupin":0,"sizes":[]})
    seen = set(); duprow = 0
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            a,_,b = line.partition("\t"); b=b.rstrip("\n")
            c = country.get(a,"?"); e = st[c]
            e["rows"] += 1
            if a in seen: duprow += 1
            seen.add(a)
            if not b.strip(): e["empty"] += 1; e["sizes"].append(0); continue
            parts=[x for x in b.split(",") if x]
            if len(set(parts))!=len(parts): e["dupin"] += 1
            e["ids"]+=len(parts); e["sizes"].append(len(parts))
            for x in parts:
                if x.startswith("S2-"): e["s2"]+=1
                elif x.startswith("S3-"): e["s3"]+=1
                if x not in valid: e["inv"]+=1
    return st, duprow

for path,label in ((f"{OUT}/matching_results.tsv","MATCHING RESULTS"),
                   (f"{OUT}/candidate_pairs.tsv","CANDIDATE PAIRS")):
    st,duprow = scan(path,None)
    print("="*104); print(label); print("="*104)
    print(f"  {'country':<9}{'rows':>10}{'empty%':>9}{'mean/S1':>9}{'p50':>6}{'p95':>7}{'p99':>7}"
          f"{'max':>7}{'S2':>11}{'S3':>11}{'invalid':>9}{'dupIDs':>8}")
    # Iterate whatever countries are actually present. Hard-coding
    # {US, India, France} would silently drop an unexpected label -- the exact
    # open-set failure the problem statement warns about, and it hid 200
    # synthetic "Zephyria" rows when this script was dry-run.
    for c in sorted(st):
        e=st[c]; z=sorted(e["sizes"]); n=len(z)
        q=lambda p: z[min(int(p*n),n-1)]
        print(f"  {c:<9}{e['rows']:>10,}{100*e['empty']/e['rows']:>8.2f}%{e['ids']/e['rows']:>9.2f}"
              f"{q(.5):>6}{q(.95):>7}{q(.99):>7}{max(z):>7}{e['s2']:>11,}{e['s3']:>11,}"
              f"{e['inv']:>9,}{e['dupin']:>8,}")
    tot=sum(e["rows"] for e in st.values())
    zero=sum(1 for c in st for _ in ())  # placeholder, real count below
    print(f"  TOTAL rows {tot:,} (need 1,732,544) | duplicate rows {duprow}")
    z = sum(e["empty"] for e in st.values())
    if label.startswith("CANDIDATE"):
        print(f"  zero-candidate S1 entities: {z:,}  (any non-zero is a RED stop)")
    print()

    # Exclusivity: ground truth assigns each S2/S3 id to at most one S1, and
    # resolve_exclusivity enforces it. After merge this MUST be zero.
    claims = collections.Counter()
    with open(path, encoding="utf-8") as fh:
        fh.readline()
        for line in fh:
            _, _, b = line.partition("\t"); b = b.rstrip("\n")
            if b.strip():
                claims.update(x for x in b.split(",") if x)
    multi = sum(1 for v in claims.values() if v > 1)
    excess = sum(v - 1 for v in claims.values() if v > 1)
    print(f"  EXCLUSIVITY: {len(claims):,} distinct ids used | "
          f"claimed by >1 S1: {multi:,} | excess claims: {excess:,}"
          f"{'  <-- VIOLATION' if (multi and label.startswith('MATCHING')) else ''}")
    print()
