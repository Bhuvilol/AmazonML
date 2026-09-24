"""TSV reading and writing for the Business Entity Resolution challenge.

Every trap this challenge sets on I/O is handled here, once, so no caller has
to remember them:

* **Tab separator is mandatory.** Reading without ``sep="\\t"`` silently yields
  one column holding the whole line. The problem statement warns about this
  explicitly, which means people hit it.
* **No quoting.** Business names contain apostrophes and stray quote
  characters (``-- Holloway Peak Inc "Seafood"``). With default CSV quoting a
  single unbalanced ``"`` swallows rows. We disable quote handling entirely,
  matching how the organisers' validator parses.
* **Everything stays a string.** ``entity_id`` values look numeric after the
  prefix (``S1-925783039``); any numeric inference risks corrupting join keys.
* **Empty is not null.** ``matched_entity_ids`` is legitimately empty for
  singletons (5.58% of training entities). An empty list must round-trip as an
  empty string, never as ``NaN`` / the literal text ``nan``.
* **Field count is asserted, not trusted.** Dirty free-text can contain stray
  tabs, which would shift columns silently.

Scale note: the full dataset is ~2.4 GB across 7 files (up to 5.3M rows each),
against 8 GB of RAM. Readers therefore default to Polars with lazy scans and
column projection, and every reader accepts a ``country`` filter so callers can
work one country partition at a time. Matches never cross countries (verified
on 693k training pairs), so partitioning is lossless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl

logger = logging.getLogger(__name__)

SEP = "\t"
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

# Polars CSV options shared by every reader. quote_char=None is the important
# one: it disables quote processing so an unbalanced `"` cannot eat rows.
_READ_OPTS = dict(
    separator=SEP,
    quote_char=None,
    has_header=True,
    infer_schema_length=0,   # force every column to Utf8
    # Empty fields must stay empty strings, never null: an empty
    # matched_entity_ids is a real singleton, not missing data.
    empty_string_is_null=False,
    encoding="utf8",
)


def scan_source(path: str | Path, country: str | None = None) -> pl.LazyFrame:
    """Lazily scan a source TSV, optionally restricted to one country.

    Returns a LazyFrame so callers can project columns and filter before any
    data is materialised -- the difference between fitting in 8 GB and not.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    frame = pl.scan_csv(path, **_READ_OPTS)
    if country is not None:
        frame = frame.filter(pl.col("country") == country)
    return frame


def read_source(
    path: str | Path,
    country: str | None = None,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Read a source TSV into memory as all-Utf8 columns.

    Args:
        path: the ``*_source{1,2,3}.tsv`` file.
        country: optional country partition filter.
        columns: optional column projection; defaults to all four.

    Raises:
        ValueError: if the header is not exactly the expected four columns,
            which is the signature of a wrong separator or a corrupted file.
    """
    frame = scan_source(path, country)
    header = frame.collect_schema().names()
    _assert_header(header, SOURCE_COLUMNS, Path(path).name)

    if columns is not None:
        frame = frame.select(list(columns))

    table = frame.collect()
    logger.info(
        "read %s: %d rows%s",
        Path(path).name, table.height, f" (country={country})" if country else "",
    )
    return table


def _assert_header(actual: list[str], expected: list[str], name: str) -> None:
    """Fail loudly on a header mismatch rather than producing garbage rows."""
    cleaned = [c.strip() for c in actual]
    if cleaned == expected:
        return
    if len(cleaned) == 1:
        raise ValueError(
            f"{name}: parsed a single column {cleaned!r}. This is the classic "
            f"symptom of reading a .tsv without an explicit tab separator."
        )
    raise ValueError(f"{name}: unexpected header {cleaned!r}; expected {expected!r}")


def iter_ground_truth(path: str | Path) -> Iterator[tuple[str, set[str]]]:
    """Stream ``(source1_entity_id, {matched ids})`` from the ground-truth TSV.

    Streams rather than loading because the training ground truth holds 2.2M
    entities and 7.6M ids -- materialising it as Python sets costs well over a
    gigabyte. Callers that genuinely need it all should use
    :func:`read_ground_truth` and accept the cost.

    An empty ``matched_entity_ids`` yields an empty set, which is a real
    singleton and not a parse failure.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(SEP)
        _assert_header(header, GT_COLUMNS, path.name)

        for line_number, line in enumerate(handle, start=2):
            entity_id, tab, rest = line.partition(SEP)
            if not tab:
                if line.strip():
                    raise ValueError(
                        f"{path.name}: malformed row (no tab) at line {line_number}: "
                        f"{line.rstrip()!r}"
                    )
                continue
            rest = rest.rstrip("\n")
            matched = {piece for piece in rest.split(",") if piece} if rest.strip() else set()
            yield entity_id.strip(), matched


def read_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """Load the whole ground truth as ``{s1_id: {matched ids}}``.

    Costs roughly 1-2 GB on the full training file. Prefer
    :func:`iter_ground_truth` unless random access is genuinely needed.
    """
    truths = dict(iter_ground_truth(path))
    singletons = sum(1 for ids in truths.values() if not ids)
    logger.info(
        "read %s: %d entities, %d singletons (%.2f%%)",
        Path(path).name, len(truths), singletons,
        100 * singletons / len(truths) if truths else 0.0,
    )
    return truths


def read_entity_ids(path: str | Path) -> list[str]:
    """Read just the first column of a source file, in file order.

    Used to build submissions: every Source-1 test entity needs exactly one
    row, so the id list defines the required output rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        handle.readline()  # header
        for line in handle:
            if not line.strip():
                continue
            ids.append(line.split(SEP, 1)[0].strip())
    logger.info("read %s: %d entity ids", path.name, len(ids))
    return ids
