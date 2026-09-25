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

rm -f Lassi_submission.zip
( cd "$STAGE" && zip -qr - Lassi_submission ) > Lassi_submission.zip

echo
echo "=== package contents ==="
unzip -l Lassi_submission.zip | tail -n +4 | head -30
echo
echo "built: $PWD/Lassi_submission.zip  ($(du -h Lassi_submission.zip | cut -f1))"
