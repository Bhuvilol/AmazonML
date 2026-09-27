"""Pairwise similarity features for the matching model.

Scale drives every choice here. The full test candidate set is ~65M pairs, so a
feature that costs one Python-level operation per pair costs 65M operations.
Two consequences:

1. **String comparisons go through ``rapidfuzz.process.cpdist``**, which is C++
   and releases the GIL (``workers=-1`` uses every core). A Python loop over
   65M pairs would take hours.
2. **Anything that depends only on a single record is precomputed once per
   record and indexed by the pair arrays.** Lengths, flags and token counts are
   computed over ~12M records, not ~65M pairs, then gathered with NumPy fancy
   indexing. Computing them per-pair would be 5x the work for identical output.

Feature choices follow the measured properties of this dataset rather than a
generic similarity checklist:

* **Order-insensitive scorers are weighted heavily** (``token_set``,
  ``token_sort``). The problem statement promises word-order transpositions and
  address component reordering, and normalize.py already sorts tokens, so
  order-sensitive scorers would mostly measure noise.
* **Numeric overlap gets its own features.** Street numbers and postal codes
  are the most selective tokens in a free-text address and, unlike words,
  survive transliteration completely intact -- making them the main bridge for
  the 11.63% of true pairs whose Source-2/3 name is non-Latin.
* **Script flags are features, not filters.** Source 1 is 100% ASCII while
  Source 2/3 are 11-15% non-Latin. For those pairs every name-similarity
  feature is structurally ~0 and carries no information. Passing the flag lets
  the model learn "ignore name similarity, trust the address" as a regime
  rather than being misled by a meaningless zero.
* **No country feature.** Matches never cross countries (verified on 693k
  pairs) and the pipeline partitions by country, so it would be constant --
  and one-hotting it is exactly what the problem statement warns against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    # --- name similarity ---
    "name_token_set",    # order-insensitive, robust to extra/missing tokens
    "name_token_sort",   # order-insensitive, sensitive to token content
    "name_ratio",        # plain Levenshtein ratio; catches typos
    "name_jaro",         # strong on short strings and shared prefixes
    "name_cos",          # TF-IDF char-n-gram cosine, reused from blocking
    # --- address similarity ---
    "addr_token_set",
    "addr_token_sort",
    "addr_ratio",
    "addr_jaro",
    "addr_cos",
    # --- numeric tokens: the transliteration-proof signal ---
    "nums_token_set",
    "nums_exact",
    "nums_both_present",
    # --- structural ---
    "name_len_ratio",
    "addr_len_ratio",
    "name_tok_diff",
    "addr_tok_diff",
    # --- regime flags ---
    "tgt_is_s3",
    "tgt_name_non_ascii",
    "tgt_addr_non_ascii",
)

# ---------------------------------------------------------------------------
# Competition features.
#
# Every feature above is *absolute*: it describes one pair in isolation. That
# leaves out what turned out to be the strongest signal available -- whether a
# pair is good RELATIVE TO THE OTHER CANDIDATES COMPETING FOR THE SAME ENTITY.
# A 0.7 name cosine means something entirely different as the best of a weak
# field than as 37th of forty strong ones, and the absolute features cannot
# express the difference.
#
# Measured A/B on 6,000 US training entities (held-out 2,400, same split and
# hyperparameters): macro F0.5 0.9800 -> 0.9894, **+0.0094** -- more than double
# any other lever tested. ``share_comb`` alone carried a LightGBM gain of
# 626,234 against 44,291 for ``addr_cos`` and 22,437 for ``addr_token_set``,
# the previous top feature.
#
# A further 8 features (second-best level, top-1 dominance, mutual-best flag,
# log target degree, mean field strength) were tested and scored +0.0092 --
# indistinguishable, so they were dropped rather than carried.
#
# These cost nothing extra to compute: everything comes from the CandidateSet
# arrays that blocking already produced.
COMPETITION_FEATURE_NAMES: tuple[str, ...] = (
    "rank_name",    # 0 = this entity's best candidate by name cosine
    "rank_addr",
    "rank_comb",
    "marg_name",    # gap to this entity's best; <= 0
    "marg_addr",
    "marg_comb",
    "rel_name",     # ratio to this entity's best; <= 1
    "rel_addr",
    "n_cands",      # how crowded the field is
    "share_comb",   # this pair's share of the entity's total similarity mass
)

# ---------------------------------------------------------------------------
# Exact-match and IDF/token features.
#
# Measured A/B on 6,000 US entities (2,400 held out, identical split and
# hyperparameters): macro F0.5 0.9800 -> 0.9894, **+0.0094**.
#
# The exact-match columns turned out to be INERT -- none reached the top-20 by
# gain, and the residual analysis found ZERO model-rejected true pairs with
# both name and address exact. They are retained because they were part of the
# validated 43-column space and removing them would change the model, not
# because they earn their place.
#
# The gain is attributable to the IDF columns: on India, `addr_idf_jaccard`
# (gain 975,400) and `addr_idf_coverage` (412,355) rank 1st and 4th overall,
# above every string-similarity feature. They answer a question no other
# feature could: whether the tokens two records SHARE are rare or ubiquitous.
IDF_FEATURE_NAMES: tuple[str, ...] = (
    "exact_name_norm",      # normalised names identical (non-empty)
    "exact_addr_norm",
    "exact_both_norm",
    "exact_name_block",     # blocking keys identical (suffix-stripped, sorted)
    "exact_addr_block",
    "name_jaccard",         # |shared tokens| / |union|
    "addr_jaccard",
    "name_idf_jaccard",     # IDF-weighted Jaccard
    "addr_idf_jaccard",
    "name_idf_coverage",    # IDF mass of query tokens covered by the target
    "addr_idf_coverage",
    "name_max_shared_idf",  # rarest shared token
    "addr_max_shared_idf",
)

N_IDF = len(IDF_FEATURE_NAMES)

ALL_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_NAMES + IDF_FEATURE_NAMES + COMPETITION_FEATURE_NAMES
)

N_FEATURES = len(FEATURE_NAMES)
N_COMPETITION = len(COMPETITION_FEATURE_NAMES)
N_ALL = len(ALL_FEATURE_NAMES)
DTYPE = np.float32


@dataclass
class RecordArrays:
    """Per-record arrays, computed once and indexed per pair.

    Build with :meth:`from_frame` so the derived columns stay consistent
    between the Source-1 side and the Source-2/3 side.
    """

    name: list[str]
    addr: list[str]
    nums: list[str]
    name_key: list[str]      # blocking key: suffix-stripped, token-sorted
    addr_key: list[str]
    name_len: np.ndarray
    addr_len: np.ndarray
    name_tokens: np.ndarray
    addr_tokens: np.ndarray
    is_s3: np.ndarray
    name_non_ascii: np.ndarray
    addr_non_ascii: np.ndarray

    @classmethod
    def from_frame(cls, frame) -> "RecordArrays":
        """Build from a Polars frame carrying normalize.py's derived columns."""
        name = frame["name_norm"].to_list()
        addr = frame["addr_norm"].to_list()
        nums = frame["addr_nums"].to_list()
        return cls(
            name=name,
            addr=addr,
            nums=nums,
            name_key=frame["name_block"].to_list(),
            addr_key=frame["addr_block"].to_list(),
            name_len=np.fromiter((len(s) for s in name), dtype=np.int32, count=len(name)),
            addr_len=np.fromiter((len(s) for s in addr), dtype=np.int32, count=len(addr)),
            name_tokens=np.fromiter(
                (s.count(" ") + 1 if s else 0 for s in name), dtype=np.int32, count=len(name)
            ),
            addr_tokens=np.fromiter(
                (s.count(" ") + 1 if s else 0 for s in addr), dtype=np.int32, count=len(addr)
            ),
            is_s3=frame["entity_id"].str.starts_with("S3-").to_numpy().astype(np.float32)
            if "entity_id" in frame.columns
            else np.zeros(len(name), dtype=np.float32),
            name_non_ascii=frame["name_non_ascii"].to_numpy().astype(np.float32),
            addr_non_ascii=frame["addr_non_ascii"].to_numpy().astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.name)


def _gather(values: list[str], index: np.ndarray) -> list[str]:
    """Gather strings by index. Kept separate so it is easy to profile."""
    return [values[i] for i in index]


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ratio of the smaller to the larger, 0 when both are 0.

    A symmetric length ratio rather than a raw difference, so the feature is
    scale-free: two 8-character names differing by 2 should look like two
    80-character names differing by 20.
    """
    hi = np.maximum(a, b).astype(DTYPE)
    lo = np.minimum(a, b).astype(DTYPE)
    out = np.zeros_like(hi, dtype=DTYPE)
    np.divide(lo, hi, out=out, where=hi > 0)
    return out


def compute_pair_features(
    left: RecordArrays,
    right: RecordArrays,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    name_cos: np.ndarray,
    addr_cos: np.ndarray,
    workers: int = -1,
) -> np.ndarray:
    """Compute the feature matrix for one chunk of candidate pairs.

    Args:
        left: Source-1 record arrays.
        right: Source-2/3 record arrays.
        left_idx, right_idx: parallel index arrays defining the pairs.
        name_cos, addr_cos: blocking cosine similarities for those pairs.
        workers: threads for rapidfuzz; -1 uses all cores.

    Returns:
        ``(n_pairs, N_FEATURES)`` float32 array, columns ordered as
        :data:`FEATURE_NAMES`.
    """
    n = len(left_idx)
    if n != len(right_idx):
        raise ValueError(f"index arrays differ in length: {n} vs {len(right_idx)}")

    ln = _gather(left.name, left_idx)
    rn = _gather(right.name, right_idx)
    la = _gather(left.addr, left_idx)
    ra = _gather(right.addr, right_idx)
    lnum = _gather(left.nums, left_idx)
    rnum = _gather(right.nums, right_idx)

    def pair_score(queries, choices, scorer) -> np.ndarray:
        # score_multiplier=0.01 puts rapidfuzz's 0-100 scores on a 0-1 scale,
        # matching the cosine features so the model sees one consistent range.
        return process.cpdist(
            queries, choices, scorer=scorer, workers=workers,
            dtype=np.float32, score_multiplier=0.01,
        )

    out = np.empty((n, N_FEATURES), dtype=DTYPE)

    out[:, 0] = pair_score(ln, rn, fuzz.token_set_ratio)
    out[:, 1] = pair_score(ln, rn, fuzz.token_sort_ratio)
    out[:, 2] = pair_score(ln, rn, fuzz.ratio)
    out[:, 3] = process.cpdist(
        ln, rn, scorer=JaroWinkler.normalized_similarity,
        workers=workers, dtype=np.float32,
    )
    out[:, 4] = name_cos

    out[:, 5] = pair_score(la, ra, fuzz.token_set_ratio)
    out[:, 6] = pair_score(la, ra, fuzz.token_sort_ratio)
    out[:, 7] = pair_score(la, ra, fuzz.ratio)
    out[:, 8] = process.cpdist(
        la, ra, scorer=JaroWinkler.normalized_similarity,
        workers=workers, dtype=np.float32,
    )
    out[:, 9] = addr_cos

    out[:, 10] = pair_score(lnum, rnum, fuzz.token_set_ratio)
    # Exact equality of the sorted digit-token string. Cheap, and a very strong
    # positive signal: identical street number and postcode is hard to hit by
    # chance, and digits survive transliteration unchanged.
    lnum_arr = np.array(lnum, dtype=object)
    rnum_arr = np.array(rnum, dtype=object)
    both_present = (lnum_arr != "") & (rnum_arr != "")
    out[:, 11] = ((lnum_arr == rnum_arr) & both_present).astype(DTYPE)
    out[:, 12] = both_present.astype(DTYPE)

    out[:, 13] = _safe_ratio(left.name_len[left_idx], right.name_len[right_idx])
    out[:, 14] = _safe_ratio(left.addr_len[left_idx], right.addr_len[right_idx])
    out[:, 15] = np.abs(
        left.name_tokens[left_idx] - right.name_tokens[right_idx]
    ).astype(DTYPE)
    out[:, 16] = np.abs(
        left.addr_tokens[left_idx] - right.addr_tokens[right_idx]
    ).astype(DTYPE)

    out[:, 17] = right.is_s3[right_idx]
    out[:, 18] = right.name_non_ascii[right_idx]
    out[:, 19] = right.addr_non_ascii[right_idx]

    return out


def iter_feature_chunks(
    left: RecordArrays,
    right: RecordArrays,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    name_cos: np.ndarray,
    addr_cos: np.ndarray,
    chunk_size: int = 2_000_000,
    workers: int = -1,
):
    """Yield ``(start, stop, features)`` over the pair arrays in chunks.

    At 65M pairs the full feature matrix is ~5 GB in float32, which does not fit
    alongside everything else in 8 GB of RAM. Chunking keeps peak usage to
    roughly ``chunk_size * N_FEATURES * 4`` bytes -- about 160 MB at the default
    -- so inference streams instead of materialising.
    """
    total = len(left_idx)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        features = compute_pair_features(
            left, right,
            left_idx[start:stop], right_idx[start:stop],
            name_cos[start:stop], addr_cos[start:stop],
            workers=workers,
        )
        logger.debug("features %d-%d / %d", start, stop, total)
        yield start, stop, features


def compute_competition_features(
    row: np.ndarray,
    col: np.ndarray,
    name_cos: np.ndarray,
    addr_cos: np.ndarray,
    n_entities: int,
    n_targets: int,
) -> np.ndarray:
    """Describe each pair relative to the others competing for the same entity.

    See :data:`COMPETITION_FEATURE_NAMES` for why this matters and what it is
    worth (+0.0094 macro F0.5, the largest single measured gain).

    This must be computed over an entity's **whole** candidate list, so callers
    pass the complete arrays for a partition, never an arbitrary slice of them:
    a slice that cuts an entity in half silently produces wrong ranks and
    shares. :func:`pipeline.score_candidates` chunks the *feature* computation
    but calls this once, up front, on everything.

    Args:
        row, col: parallel entity / target index arrays for every candidate.
        name_cos, addr_cos: blocking cosines for those pairs.
        n_entities, n_targets: partition sizes, for the bincount extents.

    Returns:
        ``(n_pairs, N_COMPETITION)`` float32, ordered as
        :data:`COMPETITION_FEATURE_NAMES`.
    """
    if not (len(row) == len(col) == len(name_cos) == len(addr_cos)):
        raise ValueError("competition inputs must be parallel arrays")

    comb = name_cos + addr_cos

    # Per-entity aggregates.
    best_n = np.zeros(n_entities, dtype=np.float32)
    best_a = np.zeros(n_entities, dtype=np.float32)
    best_c = np.zeros(n_entities, dtype=np.float32)
    np.maximum.at(best_n, row, name_cos)
    np.maximum.at(best_a, row, addr_cos)
    np.maximum.at(best_c, row, comb)
    count = np.bincount(row, minlength=n_entities).astype(np.float32)
    total = np.bincount(row, weights=comb, minlength=n_entities).astype(np.float32)

    # NOTE: tgt_degree / tgt_margin were measured and then REMOVED. They are
    # target-global, so their denominator is the partition being processed --
    # 30k entities in training versus 160k-660k in an inference shard. That is
    # a train/serve skew, not leakage. Experiment 2 validated the ten S1-local
    # features below WITHOUT them (+0.0053 US / +0.0065 India on the locked
    # holdout at a fixed threshold), so they stay out.

    def rank_within(values: np.ndarray) -> np.ndarray:
        """0-based descending rank of each pair inside its entity's list."""
        order = np.lexsort((-values, row))
        ranks = np.empty(len(values), dtype=np.float32)
        starts = np.searchsorted(row[order], np.arange(n_entities))
        ranks[order] = np.arange(len(values)) - starts[row[order]]
        return ranks

    eps = np.float32(1e-6)
    out = np.empty((len(row), N_COMPETITION), dtype=DTYPE)
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


def build_idf(texts: list[str]) -> dict[str, float]:
    """Word-level IDF over the TARGET corpus -- the same corpus blocking fits.

    ``log(N / (1 + df)) + 1``, matching sklearn's smoothed form. Fitting on the
    targets rather than the sources is deliberate and mirrors
    :func:`blocking._vectorize`: the target side is the larger, more stable
    corpus, and it is what inference sees too, so train and serve agree.
    """
    import collections
    import math

    df: collections.Counter = collections.Counter()
    for text in texts:
        if text:
            df.update(set(text.split()))
    n = len(texts)
    return {token: math.log(n / (1 + count)) + 1.0 for token, count in df.items()}


def compute_idf_features(
    left: "RecordArrays",
    right: "RecordArrays",
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    idf_name: dict[str, float],
    idf_addr: dict[str, float],
) -> np.ndarray:
    """Exact-match and IDF/token overlap features for one chunk of pairs.

    Unlike :func:`compute_competition_features` this is a pure per-pair
    computation, so it may be chunked freely.

    Returns ``(n_pairs, N_IDF)`` float32, ordered as :data:`IDF_FEATURE_NAMES`.
    """
    qn = _gather(left.name, left_idx)
    tn = _gather(right.name, right_idx)
    qa = _gather(left.addr, left_idx)
    ta = _gather(right.addr, right_idx)
    qnb = _gather(left.name_key, left_idx)
    tnb = _gather(right.name_key, right_idx)
    qab = _gather(left.addr_key, left_idx)
    tab = _gather(right.addr_key, right_idx)

    n = len(left_idx)
    out = np.zeros((n, N_IDF), dtype=DTYPE)
    gn, ga = idf_name.get, idf_addr.get

    for i in range(n):
        a_n = set(qn[i].split()) if qn[i] else set()
        b_n = set(tn[i].split()) if tn[i] else set()
        a_a = set(qa[i].split()) if qa[i] else set()
        b_a = set(ta[i].split()) if ta[i] else set()

        en = 1.0 if (qn[i] and qn[i] == tn[i]) else 0.0
        ea = 1.0 if (qa[i] and qa[i] == ta[i]) else 0.0
        out[i, 0] = en
        out[i, 1] = ea
        out[i, 2] = 1.0 if (en and ea) else 0.0
        out[i, 3] = 1.0 if (qnb[i] and qnb[i] == tnb[i]) else 0.0
        out[i, 4] = 1.0 if (qab[i] and qab[i] == tab[i]) else 0.0

        sh_n, un_n = a_n & b_n, a_n | b_n
        sh_a, un_a = a_a & b_a, a_a | b_a
        out[i, 5] = len(sh_n) / len(un_n) if un_n else 0.0
        out[i, 6] = len(sh_a) / len(un_a) if un_a else 0.0
        if un_n:
            ws = sum(gn(t, 1.0) for t in sh_n)
            out[i, 7] = ws / sum(gn(t, 1.0) for t in un_n)
            qw = sum(gn(t, 1.0) for t in a_n)
            out[i, 9] = ws / qw if qw else 0.0
            out[i, 11] = max((gn(t, 1.0) for t in sh_n), default=0.0)
        if un_a:
            ws = sum(ga(t, 1.0) for t in sh_a)
            out[i, 8] = ws / sum(ga(t, 1.0) for t in un_a)
            qw = sum(ga(t, 1.0) for t in a_a)
            out[i, 10] = ws / qw if qw else 0.0
            out[i, 12] = max((ga(t, 1.0) for t in sh_a), default=0.0)
    return out
