"""Candidate generation (blocking) -- the stage that sets the recall ceiling.

The problem statement says it directly: blocking "determines the upper bound of
your recall". The test set has 1.73M Source-1 entities against ~10M Source-2/3
records, i.e. ~1.7e13 possible pairs, so exhaustive comparison is impossible and
whatever this stage misses is lost permanently. Nothing downstream can recover a
pair that was never proposed.

Three measured facts from the training data drive the design:

1. **Matches never cross countries.** 693k training pairs checked, zero
   cross-country. Country is therefore a lossless hard partition, which shrinks
   the problem to three independent, smaller ones and keeps peak memory inside
   8 GB.
2. **Name-only blocking would concede ~14% of recall.** 13.89% of true pairs
   have a Source-2/3 name in a non-Latin script (Devanagari, Tamil) while
   Source 1 is 100% ASCII -- character n-grams score exactly 0.0 across that
   boundary. Addresses are mostly Latin even for those rows, so the candidate
   set is the **union** of name-based and address-based top-k. That union is
   what recovers the 11.63% of pairs with a non-Latin name but a Latin address.
3. **Only 2.26% of pairs have no Latin bridge at all** (both name and address
   non-Latin). That is the irreducible hole; transliteration was rejected as a
   poor use of a 48-hour budget for a 2.26% pair-level gain that macro
   averaging dilutes further.

Representation: candidates are returned as a SciPy CSR matrix of
``(n_source1, n_targets)`` holding similarity scores, not as Python sets of id
strings. At 1.73M entities and ~40 candidates each that is ~69M pairs; as
``int32`` indices it costs a few hundred MB, while Python string sets would not
fit in memory at all. Ids are resolved only at write time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

logger = logging.getLogger(__name__)

# Character n-grams, not word tokens. This is the single most important choice
# in the module: char n-grams are script- and language-agnostic, survive typos,
# and -- combined with the sorted-token blocking keys from normalize.py -- are
# insensitive to the word-order transpositions the problem statement promises.
# Word tokens would break on all three.
NGRAM_RANGE = (2, 3)
ANALYZER = "char_wb"

# float32 halves the memory of every similarity matrix versus float64 at no
# meaningful cost in ranking quality.
DTYPE = np.float32


@dataclass
class CandidateSet:
    """Candidate pairs in flat index form, with both blocking scores retained.

    Stored as parallel arrays rather than a dict of id strings: at ~65M pairs,
    ``int32`` indices cost a few hundred MB while Python string sets would not
    fit in 8 GB at all. Ids are resolved only when writing output.

    ``name_cos`` and ``addr_cos`` are kept *separately* rather than merged into
    one score. A pair proposed only by address (the cross-script case, 11.63% of
    true pairs) has ``name_cos == 0``, and that zero is informative -- it tells
    the model which regime the pair is in. Collapsing them to a max would
    destroy exactly the signal that distinguishes "names disagree" from "names
    are in different scripts".
    """

    row: np.ndarray        # source-side index, one entry per pair
    col: np.ndarray        # target-side index, one entry per pair
    name_cos: np.ndarray   # name TF-IDF cosine (0.0 if not proposed by name)
    addr_cos: np.ndarray   # address TF-IDF cosine (0.0 if not proposed by addr)
    indptr: np.ndarray     # CSR row boundaries, for per-entity grouping
    n_source: int
    n_target: int

    def __len__(self) -> int:
        return len(self.row)

    def row_slice(self, source_index: int) -> slice:
        """Slice into the flat arrays for one source entity's candidates."""
        return slice(int(self.indptr[source_index]), int(self.indptr[source_index + 1]))


@dataclass
class BlockingStats:
    """Diagnostics for the methodology write-up and for tuning ``top_n``."""

    n_source1: int = 0
    n_targets: int = 0
    n_candidates: int = 0
    name_candidates: int = 0
    addr_candidates: int = 0
    empty_rows: int = 0
    per_entity_mean: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def reduction_ratio(self) -> float:
        """Fraction of the full pair space eliminated.

        Reported by the organisers as a blocking-quality measure, so we compute
        it ourselves rather than let them discover it first.
        """
        full = self.n_source1 * self.n_targets
        return 1.0 - (self.n_candidates / full) if full else 0.0

    def describe(self) -> str:
        return (
            f"{self.n_source1:,} x {self.n_targets:,} -> {self.n_candidates:,} candidates "
            f"({self.per_entity_mean:.1f}/entity, {self.empty_rows:,} empty rows, "
            f"reduction ratio {self.reduction_ratio:.8f})"
        )


def _vectorize(
    fit_texts: list[str],
    transform_texts: list[str],
    min_df: int = 2,
    max_df: float = 0.1,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    max_features: int | None = None,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Fit TF-IDF on the target corpus and transform both sides.

    The vectoriser is fitted on the *target* side (Source 2/3) because it is the
    larger corpus and gives more stable IDF weights. Both sides must share one
    vocabulary for the dot product to be a cosine similarity at all.

    ``max_df`` is the single most important performance knob in this module.
    The top-k sparse multiply costs, per source row, roughly the sum of the
    posting-list lengths of that row's n-grams. Very common n-grams have posting
    lists in the hundreds of thousands yet near-zero IDF, so they dominate
    runtime while contributing almost nothing to the similarity. Dropping them
    is close to free in quality terms and worth an order of magnitude in speed.
    """
    vectorizer = TfidfVectorizer(
        analyzer=ANALYZER,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype=DTYPE,
        lowercase=False,   # normalize.py already lowercased
    )
    target_matrix = vectorizer.fit_transform(fit_texts)
    source_matrix = vectorizer.transform(transform_texts)
    logger.info(
        "    vocab=%d, target nnz/row=%.1f, source nnz/row=%.1f",
        len(vectorizer.vocabulary_),
        target_matrix.nnz / max(target_matrix.shape[0], 1),
        source_matrix.nnz / max(source_matrix.shape[0], 1),
    )
    return source_matrix.tocsr(), target_matrix.tocsr()


def _topn_similarity(
    source: sp.csr_matrix,
    target: sp.csr_matrix,
    top_n: int,
    threshold: float,
    chunk_rows: int,
    n_threads: int,
) -> sp.csr_matrix:
    """Top-``top_n`` cosine similarities per source row, computed in chunks.

    ``sp_matmul_topn`` keeps only the largest ``top_n`` values per row *during*
    the multiply, so the full product is never materialised -- the whole reason
    this is tractable at 810k x 4.7M.

    Source rows are chunked so the transient result stays small; the transposed
    target matrix is built once and reused across chunks, since converting it is
    the expensive part.
    """
    target_t = target.T.tocsr()
    blocks = []
    for start in range(0, source.shape[0], chunk_rows):
        block = sp_matmul_topn(
            source[start : start + chunk_rows],
            target_t,
            top_n=top_n,
            threshold=threshold,
            sort=False,
            n_threads=n_threads,
        )
        blocks.append(block)
    return sp.vstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]


def build_candidates(
    source_names: list[str],
    source_addrs: list[str],
    target_names: list[str],
    target_addrs: list[str],
    top_n_name: int = 20,
    top_n_addr: int = 20,
    min_sim_name: float = 0.25,
    min_sim_addr: float = 0.30,
    chunk_rows: int = 20_000,
    n_threads: int | None = None,
    max_df: float = 0.1,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
) -> tuple[sp.csr_matrix, BlockingStats]:
    """Generate the candidate set for one country partition.

    Returns a ``(n_source, n_target)`` CSR matrix whose stored values are the
    best similarity found for that pair (name or address), plus diagnostics.

    The union is taken with an element-wise maximum, so a pair proposed by
    either signal survives with its stronger score. Union rather than
    intersection is deliberate: this stage is recall-greedy, and precision is
    bought later by the scorer and the abstain threshold, where the F_0.5
    asymmetry is handled explicitly.
    """
    stats = BlockingStats(n_source1=len(source_names), n_targets=len(target_names))
    # -1 uses all available cores. NOTE: sparse_dot_topn does `n_threads or 1`,
    # so passing 0 or None silently runs SERIAL -- a 7x loss on this machine.
    n_threads = -1 if n_threads is None else n_threads

    logger.info(
        "blocking: %d source x %d target records", stats.n_source1, stats.n_targets
    )

    source_nm, target_nm = _vectorize(
        target_names, source_names, max_df=max_df, ngram_range=ngram_range
    )
    name_sim = _topn_similarity(
        source_nm, target_nm, top_n_name, min_sim_name, chunk_rows, n_threads
    )
    stats.name_candidates = name_sim.nnz
    logger.info("  name blocking : %d candidates", name_sim.nnz)
    del source_nm, target_nm

    source_ad, target_ad = _vectorize(
        target_addrs, source_addrs, max_df=max_df, ngram_range=ngram_range
    )
    addr_sim = _topn_similarity(
        source_ad, target_ad, top_n_addr, min_sim_addr, chunk_rows, n_threads
    )
    stats.addr_candidates = addr_sim.nnz
    logger.info("  addr blocking : %d candidates", addr_sim.nnz)
    del source_ad, target_ad

    # Union the two candidate sources while keeping both scores aligned.
    #
    # NOTE: the obvious trick of adding a zero-valued union-pattern matrix does
    # NOT work -- SciPy prunes explicit zeros during sparse addition, so the
    # result collapses back to the addend's own pattern. Instead we encode each
    # (row, col) as a sorted int64 key and gather by binary search, which is
    # exact and fully vectorised.
    union = name_sim + addr_sim
    union.sort_indices()
    row_counts = np.diff(union.indptr)
    union_rows = np.repeat(np.arange(union.shape[0], dtype=np.int64), row_counts)
    stride = np.int64(union.shape[1])
    union_keys = union_rows * stride + union.indices

    def align(sub: sp.csr_matrix) -> np.ndarray:
        """Re-express ``sub``'s values on the union's sparsity pattern."""
        sub.sort_indices()
        sub_rows = np.repeat(
            np.arange(sub.shape[0], dtype=np.int64), np.diff(sub.indptr)
        )
        sub_keys = sub_rows * stride + sub.indices
        positions = np.searchsorted(union_keys, sub_keys)
        out = np.zeros(len(union_keys), dtype=DTYPE)
        out[positions] = sub.data
        return out

    name_aligned = align(name_sim)
    addr_aligned = align(addr_sim)

    candidates = CandidateSet(
        row=union_rows.astype(np.int32),
        col=union.indices.astype(np.int32),
        name_cos=name_aligned,
        addr_cos=addr_aligned,
        indptr=union.indptr.astype(np.int64),
        n_source=union.shape[0],
        n_target=union.shape[1],
    )
    # Every parallel array must be the same length; a mismatch here silently
    # corrupts every downstream feature.
    assert len(candidates.row) == len(candidates.col) == len(candidates.name_cos) \
        == len(candidates.addr_cos) == union.nnz, "candidate arrays misaligned"

    stats.n_candidates = len(candidates)
    stats.empty_rows = int((row_counts == 0).sum())
    stats.per_entity_mean = float(row_counts.mean()) if len(row_counts) else 0.0
    logger.info("  union         : %s", stats.describe())
    return candidates, stats


def recall_ceiling(
    candidates: CandidateSet,
    truth_rows: np.ndarray,
    truth_cols: np.ndarray,
) -> dict[str, float]:
    """Fraction of true pairs present in the candidate set.

    This is the hard upper bound on achievable recall and therefore the most
    important single number produced by the pipeline -- no scorer can exceed it.
    Computed by checking membership of each true ``(row, col)`` in the candidate
    sparsity pattern.

    Args:
        candidates: the generated :class:`CandidateSet`.
        truth_rows: source-side index of each true pair.
        truth_cols: target-side index of each true pair.
    """
    if len(truth_rows) == 0:
        return {"true_pairs": 0, "recovered": 0, "recall_ceiling": float("nan")}

    # Encode each (row, col) pair as a single int64 key so membership becomes
    # one sorted binary search over a flat array. A Python loop over the 7.6M
    # true pairs would take minutes; this is a few hundred milliseconds. A dense
    # mask is impossible -- the pair space is ~1.7e13 cells.
    stride = np.int64(candidates.n_target)
    cand_keys = np.sort(candidates.row.astype(np.int64) * stride + candidates.col)
    truth_keys = truth_rows.astype(np.int64) * stride + truth_cols.astype(np.int64)

    pos = np.searchsorted(cand_keys, truth_keys)
    np.clip(pos, 0, max(len(cand_keys) - 1, 0), out=pos)
    found = cand_keys[pos] == truth_keys if len(cand_keys) else np.zeros(len(truth_keys), bool)

    recovered = int(found.sum())
    total = len(truth_rows)
    return {
        "true_pairs": total,
        "recovered": recovered,
        "recall_ceiling": recovered / total,
    }
