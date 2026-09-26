#!/bin/bash
# Assemble Lassi_submission.zip exactly as the problem statement specifies.
#
#   Lassi_submission.zip
#   ├── output/{matching_results,candidate_pairs}.tsv
#   ├── code/business_entity_resolution/{src/,README.md,requirements.txt}
#   └── Documentation_template.md
#
# Takes the output directory as $1 so the v2 results in output_v2/ can be
# packaged without first overwriting the v1 files. Refuses to build a zip that
# would be rejected: every check the organisers' validator makes is run first,
# because a package that fails validation is not evaluated at all.

set -euo pipefail
cd "$(dirname "$0")"
SRCOUT="${1:-output_v2}"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

PKG="$STAGE/Lassi_submission"
mkdir -p "$PKG/output" "$PKG/code"

for f in matching_results.tsv candidate_pairs.tsv; do
  [ -f "$SRCOUT/$f" ] || { echo "MISSING: $SRCOUT/$f"; exit 1; }
done

echo "=== validating before packaging ==="
python3 student_resource/utils/validate_submission.py \
  --matching "$SRCOUT/matching_results.tsv" \
  --candidate "$SRCOUT/candidate_pairs.tsv" \
  --test-dir student_resource/dataset/test --check-ids || {
    echo "REFUSING to build a zip from files that fail validation."; exit 1; }

cp "$SRCOUT/matching_results.tsv" "$SRCOUT/candidate_pairs.tsv" "$PKG/output/"
cp -R Lassi_submission/code/business_entity_resolution "$PKG/code/"
cp Lassi_submission/Documentation_template.md "$PKG/"

# Caches are noise in a package that gets read by a human reviewer.
find "$PKG" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find "$PKG" \( -name '*.pyc' -o -name '.DS_Store' \) -delete 2>/dev/null || true

for required in \
    "$PKG/output/matching_results.tsv" \
    "$PKG/output/candidate_pairs.tsv" \
    "$PKG/code/business_entity_resolution/src/pipeline.py" \
    "$PKG/code/business_entity_resolution/README.md" \
    "$PKG/code/business_entity_resolution/requirements.txt" \
    "$PKG/Documentation_template.md"; do
  [ -e "$required" ] || { echo "MISSING from package: $required"; exit 1; }
done

# Write to a real path inside the staging dir, then move it into place. Piping
# `zip -r -` to stdout produces a stream-format archive with no usable central
# directory: the file looks right and has a plausible size, but unzip reports
# "End-of-central-directory signature not found". That produced a 187 MB
# unreadable submission package that passed every check except opening it.
rm -f Lassi_submission.zip
( cd "$STAGE" && zip -qr Lassi_submission.zip Lassi_submission )
mv "$STAGE/Lassi_submission.zip" Lassi_submission.zip

# Never hand over an archive without opening it.
echo
echo "=== verifying archive ==="
unzip -t Lassi_submission.zip > /dev/null || { echo "ARCHIVE IS CORRUPT"; exit 1; }
# List ONCE into a variable. Piping `unzip -l` into `grep -q` per file looks
# fine and is wrong twice over: grep -q exits on its first match, which SIGPIPEs
# unzip, and under `set -o pipefail` that makes the pipeline report failure for
# a file that IS present. It reported candidate_pairs.tsv missing from an
# archive containing it, purely because that name sorts early in the listing.
# (It also re-read a 757 MB archive once per file.)
LISTING=$(unzip -l Lassi_submission.zip)
N=$(tail -1 <<<"$LISTING" | awk '{print $2}')
echo "  integrity OK, $N files"
for required in Lassi_submission/output/matching_results.tsv \
                Lassi_submission/output/candidate_pairs.tsv \
                Lassi_submission/Documentation_template.md \
                Lassi_submission/code/business_entity_resolution/requirements.txt; do
  grep -qF "$required" <<<"$LISTING" \
    || { echo "  MISSING IN ARCHIVE: $required"; exit 1; }
  echo "  present: $required"
done

echo
echo "=== package contents ==="
unzip -l Lassi_submission.zip | tail -n +4 | head -26
echo
echo "built: $PWD/Lassi_submission.zip  ($(du -h Lassi_submission.zip | cut -f1))"
