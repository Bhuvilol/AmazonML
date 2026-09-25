"""Generate a tiny synthetic dataset in the exact challenge format.

Purpose is plumbing validation, not model quality: it exercises the full CLI
path (train -> per-country predict -> merge -> official validator) in seconds
instead of hours, so a bug in argument handling or file layout is found locally
rather than after a multi-hour hosted run.

Every noise pattern the problem statement names is injected deliberately:
abbreviation swaps, legal-suffix inconsistency, `&` vs `and`, word-order
transposition, typos, dropped address components, landmark references, and a
non-Latin script. It also includes a country present ONLY in test, mirroring
France, so the open-set requirement is genuinely exercised.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

SEP = "\t"

WORDS = ["apex", "summit", "harbor", "orchid", "pioneer", "vertex", "meadow",
         "granite", "lantern", "cobalt", "juniper", "falcon", "beacon", "tidal"]
KINDS = ["traders", "foods", "motors", "textiles", "clinic", "bakery", "logistics"]
SUFFIX = {"US": ["Inc", "Corp", "LLC", "Corporation"],
          "India": ["Pvt Ltd", "Private Limited", "LLP"],
          "Zephyria": ["SARL", "SA", "SAS"]}           # test-only, stands in for France
STREETS = [("Rd", "Road"), ("St", "Street"), ("Ave", "Avenue"), ("Dr", "Drive")]
CITIES = {"US": ["Fairview", "Kingsport", "Lakeside"],
          "India": ["Indrapuram", "Vasantnagar", "Surajpur"],
          "Zephyria": ["Montclair", "Beauvais", "Larimont"]}
DEVANAGARI = ["राम", "सूर्य", "गंगा", "अमृत", "चंद्र"]


def typo(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    i = rng.randrange(1, len(text) - 1)
    return text[:i] + text[i + 1] + text[i] + text[i + 2:]


def make_name(rng: random.Random, country: str) -> str:
    return f"{rng.choice(WORDS).title()} {rng.choice(KINDS).title()} {rng.choice(SUFFIX[country])}"


def make_address(rng: random.Random, country: str) -> str:
    abbr, full = rng.choice(STREETS)
    return (f"{rng.randrange(1, 9999)} {rng.choice(WORDS).title()} {full}, "
            f"{rng.choice(CITIES[country])}, {rng.randrange(10000, 99999)}")


def vary_name(name: str, country: str, rng: random.Random) -> str:
    out = name
    roll = rng.random()
    if roll < 0.18 and country == "India":
        # Non-Latin script: char n-grams score 0.0 against the Latin source.
        return f"{rng.choice(DEVANAGARI)} {rng.choice(KINDS)} {rng.choice(SUFFIX[country])}"
    if roll < 0.40:                                    # legal-suffix inconsistency
        for s in SUFFIX[country]:
            out = out.replace(s, rng.choice(SUFFIX[country]))
    if rng.random() < 0.30:                            # word-order transposition
        parts = out.split()
        rng.shuffle(parts)
        out = " ".join(parts)
    if rng.random() < 0.25:
        out = typo(out, rng)
    if rng.random() < 0.20:
        out = out.replace(" and ", " & ")
    return out


def vary_address(addr: str, rng: random.Random) -> str:
    out = addr
    for abbr, full in STREETS:                         # abbreviation variation
        if full in out and rng.random() < 0.5:
            out = out.replace(full, abbr)
    if rng.random() < 0.25:                            # drop the postal code
        out = ",".join(out.split(",")[:-1])
    if rng.random() < 0.15:                            # landmark reference
        out = f"Near {rng.choice(WORDS).title()} Mall, " + out
    return out


def build(split: str, out_dir: Path, n_entities: int, countries: list[str], seed: int) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1, s2, s3, truth = [], [], [], []
    next_id = 1000

    for i in range(n_entities):
        country = countries[i % len(countries)]
        s1_id = f"S1-{next_id}"; next_id += 1
        name, addr = make_name(rng, country), make_address(rng, country)
        s1.append((s1_id, name, addr, country))

        matched = []
        # ~8% singletons, mirroring the real 5.58%
        n_match = 0 if rng.random() < 0.08 else rng.randrange(1, 5)
        for _ in range(n_match):
            side = rng.choice(["S2", "S3"])
            mid = f"{side}-{next_id}"; next_id += 1
            row = (mid, vary_name(name, country, rng), vary_address(addr, rng), country)
            (s2 if side == "S2" else s3).append(row)
            matched.append(mid)
        truth.append((s1_id, ",".join(matched)))

        # Distractors: same city, unrelated business. These are the hard negatives.
        for _ in range(rng.randrange(1, 4)):
            side = rng.choice(["S2", "S3"])
            mid = f"{side}-{next_id}"; next_id += 1
            row = (mid, make_name(rng, country), make_address(rng, country), country)
            (s2 if side == "S2" else s3).append(row)

    def write(path: Path, header: tuple, rows: list) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(SEP.join(header) + "\n")
            for r in rows:
                fh.write(SEP.join(r) + "\n")

    cols = ("entity_id", "business_name", "business_address", "country")
    write(out_dir / f"{split}_source1.tsv", cols, s1)
    write(out_dir / f"{split}_source2.tsv", cols, s2)
    write(out_dir / f"{split}_source3.tsv", cols, s3)
    if split == "train":
        write(out_dir / "train_ground_truth.tsv",
              ("source1_entity_id", "matched_entity_ids"), truth)

    print(f"{split}: {len(s1)} S1, {len(s2)} S2, {len(s3)} S3, "
          f"countries={sorted({r[3] for r in s1})}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="dataset root to create")
    ap.add_argument("--train-entities", type=int, default=1200)
    ap.add_argument("--test-entities", type=int, default=600)
    args = ap.parse_args()

    root = Path(args.out)
    build("train", root / "train", args.train_entities, ["US", "India"], seed=1)
    # Test adds a country absent from training -- the France analogue.
    build("test", root / "test", args.test_entities, ["US", "India", "Zephyria"], seed=2)


if __name__ == "__main__":
    main()
