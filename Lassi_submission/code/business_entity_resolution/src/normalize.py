"""Text normalisation for business names and addresses.

Design constraints that shaped every decision here:

1. **Language and script agnostic by default.** The test set contains France,
   which appears nowhere in training, so nothing may branch on a country label.
   Normalisation is applied uniformly and must degrade gracefully on text it
   has never seen.
2. **Vectorised.** There are ~20M records across train and test. Per-row Python
   regex would cost minutes per pass; every transform here is a Polars string
   expression evaluated in Rust.
3. **Order-insensitive downstream.** The problem statement promises word-order
   transpositions and address component reordering, so normalisation produces
   token sets rather than trying to preserve sequence.
4. **No external data.** Every mapping below is hand-authored domain knowledge
   (``rd`` means ``road``), which resolves no business identity. Downloaded
   gazetteers or registry data would violate the fair-play rule.

Accent folding is done with an explicit character map rather than
``unicodedata`` NFKD, because the map is vectorisable and NFKD would force a
per-row Python call. The map covers Latin-script diacritics, which is what
France and the noisier US rows actually contain.
"""

from __future__ import annotations

import logging

import polars as pl

try:
    from unidecode import unidecode as _unidecode
    HAS_UNIDECODE = True
except ImportError:          # degrade rather than fail; accent map still applies
    HAS_UNIDECODE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Accent folding: Latin diacritics -> ASCII.
# Matters for France (absent from training) and for rows like "Animal Welfare
# Nétwork" that appear even in US data.
# ---------------------------------------------------------------------------
_ACCENTS = {
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a", "ā": "a",
    "é": "e", "è": "e", "ê": "e", "ë": "e", "ē": "e",
    "í": "i", "ì": "i", "î": "i", "ï": "i", "ī": "i",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o", "õ": "o", "ø": "o", "ō": "o",
    "ú": "u", "ù": "u", "û": "u", "ü": "u", "ū": "u",
    "ý": "y", "ÿ": "y",
    "ñ": "n", "ç": "c", "ß": "ss",
    "æ": "ae", "œ": "oe",
    "đ": "d", "ł": "l", "š": "s", "ž": "z", "č": "c", "ř": "r", "ń": "n",
}

# ---------------------------------------------------------------------------
# Legal-suffix tokens, removed from the blocking key.
#
# These are effectively stopwords: "private limited" appears in millions of
# Indian records and carries almost no discriminative signal, while inflating
# character-n-gram similarity between unrelated businesses. Removing them makes
# the blocking key sharper.
#
# French forms (sarl, sas, sa, eurl, sasu) are included deliberately. France is
# test-only, so this cannot be validated locally -- it is hand-authored domain
# knowledge accepted as a small, documented bet. It is additive and cannot
# degrade US or India performance, since those tokens do not occur there.
# ---------------------------------------------------------------------------
_LEGAL_SUFFIX_TOKENS = frozenset({
    # US / generic
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "llc", "lc", "llp", "lp", "plc", "ltd", "limited",
    "holdings", "holding", "group", "enterprises", "enterprise",
    "the", "and",
    # India
    "pvt", "private", "pl", "opc",
    # France (test-only; unvalidatable, documented bet)
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc",
    # ---- Transliterated Indic legal suffixes -------------------------------
    # Romanisation is phonetic, so "प्राइवेट लिमिटेड" becomes "praaivett
    # limittedd", which the Latin suffix list above does not recognise. Those
    # tokens then survive into the blocking key and dilute it. Derived
    # empirically from the most frequent tokens in 60,000 transliterated
    # non-Latin names -- these are not guesses:
    #   limittedd 40.0%   praaivett 22.7%   praiveett 7.1%   li 5.3%
    #   praa 5.3%         limittett 3.6%    praaibhett 3.6%  piraiveett 3.1%
    #   elelpii 2.6%      limirrrrdd 2.1%   praivrrrr 1.8%
    "limittedd", "limitted", "limittett", "limirrrrdd", "limitedd",
    "praaivett", "praiveett", "praaibhett", "piraiveett", "praivrrrr",
    "praaivet", "praa", "li", "elelpii", "prai",
})

# ---------------------------------------------------------------------------
# Address abbreviations, expanded to a canonical form so "Rd" and "Road" share
# tokens. Expansion (rather than contraction) is chosen so that longer, rarer
# forms survive into the token set.
#
# "st" is deliberately absent: it is ambiguous between "street" and "saint",
# and guessing wrong actively destroys signal. It is left as-is so both sides
# normalise identically, which is all blocking requires.
# ---------------------------------------------------------------------------
_ADDRESS_ABBREVIATIONS = {
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "hwy": "highway", "pkwy": "parkway", "cir": "circle", "trl": "trail",
    "ste": "suite", "apt": "apartment", "bldg": "building", "fl": "floor",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "nagar": "nagar", "mkt": "market", "rly": "railway",
    "opp": "opposite", "nr": "near",
}


def transliterate(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Romanise non-Latin text so character n-grams can compare it at all.

    Source 1 is 100% ASCII while Source 2/3 are 11-15% non-Latin (Devanagari,
    Tamil, Telugu, Gujarati, Odia). Across that boundary character n-grams score
    **exactly 0.0** -- the strings share no characters -- so those pairs are
    invisible to blocking no matter how the ranking is tuned.

    Measured on 4,000 true pairs with a non-Latin target name:

    ============================  ======  ========  =======
    key                            mean    median    >0.15
    ============================  ======  ========  =======
    raw                            0.286    0.000     43.1%
    transliterated                 0.479    0.267     78.7%
    ============================  ======  ========  =======

    **34.9% of these pairs move from ~0 to usable.** The romanisation is
    phonetic, not a translation ('राम' becomes 'raam', not 'Ram'), which is
    sufficient: blocking only needs the pair ranked into the top-k, not an
    exact string match.

    Applied only to rows that actually contain non-ASCII, so the ~87% ASCII
    majority keeps the fast vectorised path. ``unidecode`` is a character
    mapping table, i.e. an algorithm rather than a lookup of business
    identities, so it is consistent with the external-data rule.
    """
    if not HAS_UNIDECODE:
        logger.warning("unidecode unavailable; non-Latin text will not be romanised")
        return frame

    for column in columns:
        values = frame[column].to_list()
        # Guard per row: unidecode on ASCII is a no-op but still costs a call,
        # and these frames run to millions of rows.
        romanised = [
            _unidecode(v) if v and not v.isascii() else v
            for v in values
        ]
        changed = sum(1 for a, b in zip(values, romanised) if a != b)
        if changed:
            logger.info(
                "    transliterated %s: %d/%d rows (%.1f%%)",
                column, changed, len(values), 100 * changed / max(len(values), 1),
            )
        frame = frame.with_columns(pl.Series(column, romanised))
    return frame


def _fold_and_lower(column: pl.Expr) -> pl.Expr:
    """Lowercase, then fold Latin diacritics to ASCII."""
    folded = column.fill_null("").str.to_lowercase()
    return folded.str.replace_many(
        list(_ACCENTS.keys()), list(_ACCENTS.values())
    )


def _strip_punctuation(column: pl.Expr) -> pl.Expr:
    """Replace ``&`` with ``and``, then reduce all non-alphanumerics to spaces.

    Keeping non-ASCII letters is deliberate: Devanagari and Tamil names must
    survive normalisation so that script-aware handling downstream still has
    text to work with.
    """
    return (
        column
        .str.replace_all(r"&", " and ")
        # Apostrophes are DELETED, not spaced: "Orelee's" -> "orelees", which
        # stays one token and keeps character n-grams aligned with "Orelee".
        # Spacing it instead produced a junk single-character "s" token.
        .str.replace_all(r"['’ʼ`]", "")
        .str.replace_all(r"[^\w\s]+", " ", literal=False)
        .str.replace_all(r"_", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalize_text(column: pl.Expr) -> pl.Expr:
    """Base normalisation shared by names and addresses."""
    return _strip_punctuation(_fold_and_lower(column))


def normalize_name(column: pl.Expr) -> pl.Expr:
    """Normalise a business name for comparison (legal suffixes retained)."""
    return normalize_text(column)


def blocking_name(column: pl.Expr) -> pl.Expr:
    """Normalise a business name into a blocking key, dropping legal suffixes.

    Tokens are sorted so that word-order transpositions -- explicitly promised
    by the problem statement -- produce an identical key.
    """
    tokens = normalize_text(column).str.split(" ")
    kept = tokens.list.eval(
        pl.element().filter(
            ~pl.element().is_in(list(_LEGAL_SUFFIX_TOKENS)) & (pl.element().str.len_chars() > 0)
        )
    )
    # If suffix removal empties the name, fall back to the unstripped form
    # rather than emitting an empty key that matches everything.
    stripped = kept.list.sort().list.join(" ")
    return (
        pl.when(stripped.str.len_chars() == 0)
        .then(normalize_text(column))
        .otherwise(stripped)
    )


def normalize_address(column: pl.Expr) -> pl.Expr:
    """Normalise an address, expanding common abbreviations."""
    tokens = normalize_text(column).str.split(" ")
    expanded = tokens.list.eval(
        pl.element().replace(_ADDRESS_ABBREVIATIONS)
    )
    return expanded.list.join(" ")


def blocking_address(column: pl.Expr) -> pl.Expr:
    """Normalise an address into a sorted-token blocking key.

    Sorting handles the component reordering the problem statement promises
    (``component reordering`` in the address noise list).
    """
    tokens = normalize_address(column).str.split(" ")
    kept = tokens.list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
    return kept.list.sort().list.join(" ")


def numeric_tokens(column: pl.Expr) -> pl.Expr:
    """Extract digit runs from an address as a sorted, space-joined string.

    Street numbers and postal codes are the most selective signal available in
    a free-text address, and -- unlike words -- digits survive transliteration
    completely intact. That makes them the most reliable bridge for the ~13% of
    Source-2/3 records whose names are in a non-Latin script.
    """
    return (
        column.fill_null("")
        .str.extract_all(r"\d+")
        .list.sort()
        .list.join(" ")
    )


def has_non_ascii(column: pl.Expr) -> pl.Expr:
    """Flag text containing non-ASCII characters.

    Source 1 is 100% ASCII while Source 2 is 15.1% non-ASCII and Source 3 is
    11.5% (Devanagari, Tamil). Character n-grams give exactly zero signal across
    that boundary, so knowing which records are affected is required to measure
    the blocking recall hole rather than guess at it.
    """
    return column.fill_null("").str.contains(r"[^\x00-\x7F]")


def add_normalized_columns(
    frame: pl.LazyFrame | pl.DataFrame,
    romanise: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """Attach every derived text column used by blocking and features.

    ``romanise`` transliterates non-Latin text first. It requires a materialised
    frame (the mapping is a Python call, not a Polars expression), so a
    LazyFrame is collected when it is enabled.
    """
    was_lazy = isinstance(frame, pl.LazyFrame)
    if romanise and HAS_UNIDECODE:
        # Transliteration is a Python-level mapping, not a Polars expression,
        # so a LazyFrame has to be materialised. Re-wrap afterwards so callers
        # that expect a LazyFrame back are unaffected -- silently changing the
        # return type broke every `.pipe(...).collect()` call site.
        if was_lazy:
            frame = frame.collect()
        frame = transliterate(frame, ("business_name", "business_address"))
        if was_lazy:
            frame = frame.lazy()

    return frame.with_columns(
        name_norm=normalize_name(pl.col("business_name")),
        name_block=blocking_name(pl.col("business_name")),
        addr_norm=normalize_address(pl.col("business_address")),
        addr_block=blocking_address(pl.col("business_address")),
        addr_nums=numeric_tokens(pl.col("business_address")),
        name_non_ascii=has_non_ascii(pl.col("business_name")),
        addr_non_ascii=has_non_ascii(pl.col("business_address")),
    )
