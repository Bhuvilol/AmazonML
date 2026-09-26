#!/bin/bash
# Ship something NOW: v2 India (done) + v1 US/France (already on disk).
#
# India is 809,986 of 1,732,544 test entities -- 47% -- and v2 lifted India's
# blocking recall ceiling from 0.8796 to 0.9254. Waiting for the US and France
# shards to finish before uploading anything leaves a measurable improvement
# sitting unused for hours. This merges what exists into a valid submission.
set -uo pipefail
cd "$(dirname "$0")"
OUT=hybrid
SRC=Lassi_submission/code/business_entity_resolution/src

echo "=== waiting for India downloads ==="
for K in india-s0of5 india-s1of5 india-s2of5 india-s3of5 india-s4of5; do
  if [ ! -d "$OUT/raw/$K" ]; then
    .venv/bin/kaggle kernels output "zeroxbhuvii/lassi2-$K" -p "$OUT/raw/$K" >/dev/null 2>&1 \
      && echo "  got $K" || { echo "  FAILED $K"; exit 1; }
  else
    echo "  have $K"
  fi
done

rm -f "$OUT"/*.tsv
# v2 India shards.
find "$OUT/raw" -name 'matching_results_India_*.tsv' -exec cp {} "$OUT"/ \;
find "$OUT/raw" -name 'candidate_pairs_India_*.tsv'  -exec cp {} "$OUT"/ \;
NI=$(ls "$OUT"/matching_results_India_*.tsv 2>/dev/null | wc -l | tr -d ' ')
echo "  v2 India partials: $NI (expect 5)"
[ "$NI" -eq 5 ] || { echo "ABORT: expected 5 India shards, found $NI"; exit 1; }

# v1 US and France, untouched from the previous run.
for C in US France; do
  cp "Lassi_submission/output/matching_results_$C.tsv" "$OUT/" || exit 1
  cp "Lassi_submission/output/candidate_pairs_$C.tsv"  "$OUT/" || exit 1
  echo "  v1 $C carried over"
done

echo
echo "=== merging ==="
PYTHONPATH="$SRC" .venv/bin/python "$SRC/pipeline.py" merge --output-dir "$OUT" || exit 1

echo
echo "=== validating ==="
python3 student_resource/utils/validate_submission.py \
  --matching "$OUT/matching_results.tsv" \
  --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir student_resource/dataset/test --check-ids || exit 1

echo
echo "rows: $(($(wc -l < "$OUT/matching_results.tsv") - 1))  (need 1732544)"
echo
echo "UPLOAD THIS:  $PWD/$OUT/matching_results.tsv"
