#!/bin/bash
# Wait for all 9 production shards -> download -> merge -> validate.
# Writes to output_final/ (clean dir: merge_partials globs matching_results_*)
set -uo pipefail
cd "$(dirname "$0")"
KG=.venv/bin/kaggle
OUT=output_final; RAW=$OUT/raw
SRC=Lassi_submission/code/business_entity_resolution/src
K9=(india-s0of5 india-s1of5 india-s2of5 india-s3of5 india-s4of5 us-s0of3 us-s1of3 us-s2of3 france)
mkdir -p "$RAW"

echo "=== waiting for 9 shards ==="
while :; do
  D=0;R=0;P=0;E=0
  for K in "${K9[@]}"; do
    S=$($KG kernels status "zeroxbhuvii/lassifinal-$K" 2>&1 | tail -1)
    case "$S" in
      *COMPLETE*) D=$((D+1));; *RUNNING*|*QUEUED*) R=$((R+1));;
      *ERROR*|*CANCEL*) E=$((E+1)); echo "  !! $K FAILED";; *) P=$((P+1));;
    esac
  done
  echo "$(date +%H:%M:%S)  complete=$D running=$R pending=$P error=$E"
  [ $D -eq 9 ] && break
  [ $E -gt 0 ] && [ $((D+E)) -eq 9 ] && { echo "ABORT: $E failed"; exit 1; }
  sleep 180
done

echo; echo "=== downloading ==="
for K in "${K9[@]}"; do
  $KG kernels output "zeroxbhuvii/lassifinal-$K" -p "$RAW/$K" >/dev/null 2>&1 \
    && echo "  got $K" || { echo "  FAILED $K"; exit 1; }
done
rm -f "$OUT"/*.tsv
find "$RAW" -name 'matching_results_*.tsv' -exec cp {} "$OUT"/ \;
find "$RAW" -name 'candidate_pairs_*.tsv'  -exec cp {} "$OUT"/ \;
echo "  partials: $(ls "$OUT"/matching_results_*.tsv | wc -l) matching"

echo; echo "=== merging ==="
PYTHONPATH="$SRC" .venv/bin/python "$SRC/pipeline.py" merge --output-dir "$OUT" || exit 1

echo; echo "=== official validator ==="
python3 student_resource/utils/validate_submission.py \
  --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir student_resource/dataset/test --check-ids
echo; echo "UPLOAD: $PWD/$OUT/matching_results.tsv"
