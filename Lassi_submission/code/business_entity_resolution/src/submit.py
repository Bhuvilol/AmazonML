"""Submission writers that make a malformed output file impossible.

The organisers ship ``utils/validate_submission.py``, and a submission that
fails it is not evaluated at all. Rather than write files and hope, this module
enforces every rule the validator checks *at write time* and raises instead of
emitting a bad file.

Rules enforced here, each mirroring a specific check in the official validator:

===================================  ====================================
Rule                                 Why it bites
===================================  ====================================
Exact header, tab-separated          A wrong header aborts parsing outright
One row per Source-1 test entity     Missing entities cause rejection
Every row contains a tab             ``S1-1\\n`` is a "malformed row" error;
                                     an empty list still needs ``S1-1\\t``
No duplicate ``source1_entity_id``   Rejection
No duplicate ids within a list       Rejection
No ``S1-`` ids (self-matches)        Rejection
Only ``S2-``/``S3-`` prefixes        Rejection
No whitespace around ids             Ids are not stripped by the validator,
                                     so ``S2-1, S2-2`` yields ``" S2-2"``
                                     which then fails the prefix check
UTF-8 output                         A decode error fails the whole run
===================================  ====================================

Ids are written sorted so that two runs over the same data produce
byte-identical files -- reproducibility is a graded part of this challenge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

SEP = "\t"
MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")

VALID_PREFIXES = ("S2-", "S3-")
MAX_REPORTED = 5


class SubmissionError(ValueError):
    """Raised when output would violate a rule the official validator enforces."""


def _describe(items: Iterable[str]) -> str:
    """Short sample of offending ids for an error message."""
    ordered = sorted(items)
    head = ", ".join(ordered[:MAX_REPORTED])
    return f"{len(ordered)} total, e.g. {head}" if len(ordered) > MAX_REPORTED else head


def _clean_id_list(entity_id: str, raw_ids: Iterable[str]) -> list[str]:
    """Validate and canonicalise one entity's id list.

    Returns a sorted list of ids. Raises on anything the validator would reject,
    naming the offending entity so the caller can find it.
    """
    ids = [str(value).strip() for value in raw_ids]
    ids = [value for value in ids if value]

    unique = set(ids)
    if len(unique) != len(ids):
        duplicates = {value for value in ids if ids.count(value) > 1}
        raise SubmissionError(
            f"{entity_id}: duplicate id(s) within its list: {_describe(duplicates)}. "
            f"Duplicate ids inside a list cause rejection."
        )

    self_matches = {value for value in unique if value.startswith("S1-")}
    if self_matches:
        raise SubmissionError(
            f"{entity_id}: self-match to Source 1: {_describe(self_matches)}. "
            f"Only S2-/S3- ids are allowed."
        )

    bad_prefix = {value for value in unique if not value.startswith(VALID_PREFIXES)}
    if bad_prefix:
        raise SubmissionError(
            f"{entity_id}: id(s) without an S2-/S3- prefix: {_describe(bad_prefix)}."
        )

    return sorted(unique)


def write_id_list_tsv(
    path: str | Path,
    predictions: Mapping[str, Iterable[str]],
    required_ids: Sequence[str],
    header: tuple[str, str],
) -> dict[str, int]:
    """Write one results-style TSV covering exactly ``required_ids``.

    Args:
        path: destination file; parent directories are created.
        predictions: entity id -> ids to emit. Entities absent from this
            mapping are written as empty lists, which is the correct encoding
            for a singleton.
        required_ids: every Source-1 test entity, in output order. This is the
            authority on which rows must exist -- a missing row is a rejection.
        header: :data:`MATCHING_HEADER` or :data:`CANDIDATE_HEADER`.

    Returns:
        Counts of rows written, empty rows, and total ids emitted.

    Raises:
        SubmissionError: on duplicate required ids, predictions for unknown
            entities, or any per-row rule violation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(set(required_ids)) != len(required_ids):
        seen: set[str] = set()
        duplicates = {i for i in required_ids if i in seen or seen.add(i)}  # type: ignore[func-returns-value]
        raise SubmissionError(
            f"required_ids contains duplicate entity id(s): {_describe(duplicates)}. "
            f"Duplicate source1_entity_id rows cause rejection."
        )

    required_set = set(required_ids)
    unknown = set(predictions) - required_set
    if unknown:
        raise SubmissionError(
            f"predictions reference {len(unknown)} entity id(s) absent from the test "
            f"set: {_describe(unknown)}. These would be rejected as unknown rows."
        )

    rows = empties = total_ids = 0
    # newline="" keeps Python from translating "\n"; we control line endings.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(SEP.join(header) + "\n")
        for entity_id in required_ids:
            ids = _clean_id_list(entity_id, predictions.get(entity_id, ()))
            # The tab is always written -- an empty list must still produce
            # "S1-1\t", because a row without a tab is a malformed-row error.
            handle.write(f"{entity_id}{SEP}{','.join(ids)}\n")
            rows += 1
            if ids:
                total_ids += len(ids)
            else:
                empties += 1

    stats = {"rows": rows, "empty_rows": empties, "total_ids": total_ids}
    logger.info(
        "wrote %s: %d rows (%d empty, %d non-empty), %d ids",
        path.name, rows, empties, rows - empties, total_ids,
    )
    return stats


def write_matching_results(
    path: str | Path,
    predictions: Mapping[str, Iterable[str]],
    required_ids: Sequence[str],
) -> dict[str, int]:
    """Write ``matching_results.tsv`` -- the file scored on the leaderboard."""
    return write_id_list_tsv(path, predictions, required_ids, MATCHING_HEADER)


def write_candidate_pairs(
    path: str | Path,
    candidates: Mapping[str, Iterable[str]],
    required_ids: Sequence[str],
) -> dict[str, int]:
    """Write ``candidate_pairs.tsv`` -- the blocking set fed to the model.

    Not scored, but audited for recall ceiling and reduction ratio, and the
    final matches are expected to be a subset of it.
    """
    return write_id_list_tsv(path, candidates, required_ids, CANDIDATE_HEADER)


def check_subset(
    predictions: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
) -> list[str]:
    """Return entities whose matches are not a subset of their candidates.

    The official validator only *warns* about this, but it means the pipeline
    emitted something its own blocking stage never proposed -- which is a bug
    worth failing a build over, not a warning worth scrolling past.
    """
    offenders = []
    for entity_id, matched in predictions.items():
        matched_set = set(matched)
        if matched_set - set(candidates.get(entity_id, ())):
            offenders.append(entity_id)
    if offenders:
        logger.warning(
            "%d entity(ies) have matches outside their candidate set, e.g. %s",
            len(offenders), offenders[:MAX_REPORTED],
        )
    return offenders
