#!/bin/bash
# Wait for all nine v2 shards, download their outputs, merge and validate.
#
# Runs unattended: poll until every kernel is COMPLETE, pull each one's output,
# merge into a single pair of TSVs, then run the organisers' validator. Leaves
# a ready-to-upload matching_results.tsv.
#
# Output goes to output_v2/, NOT Lassi_submission/output/. That directory still
# holds the v1 partials (matching_results_India.tsv etc.) and merge_partials
# globs matching_results_*.tsv -- mixing v1 and v2 partials would produce a file
# with duplicated and half-stale entities that still passes a row count.

set -uo pipefail
cd "$(dirname "$0")"
KG=.venv/bin/kaggle
OUT=output_v2
RAW=$OUT/raw
SRC=Lassi_submission/code/business_entity_resolution/src
INTERVAL=180

KERNELS=(india-s0of5 india-s1of5 india-s2of5 india-s3of5 india-s4of5
         us-s0of3 us-s1of3 us-s2of3 france)

mkdir -p "$RAW"

echo "=== waiting for 9 shards ==="
while :; do
  DONE=0; RUN=0; MISS=0; ERR=0
  for K in "${KERNELS[@]}"; do
    S=$($KG kernels status "zeroxbhuvii/lassi2-$K" 2>&1 | tail -1)
    case "$S" in
      *COMPLETE*)            DONE=$((DONE+1)) ;;
      *RUNNING*|*QUEUED*)    RUN=$((RUN+1)) ;;
      *ERROR*|*CANCEL*)      ERR=$((ERR+1)); echo "  !! $K FAILED" ;;
      *"Authentication required"*)
          echo "$(date +%H:%M:%S)  AUTH EXPIRED -> run: kaggle auth login --force"
          MISS=$((MISS+1)) ;;
      *)                     MISS=$((MISS+1)) ;;
    esac
  done
  echo "$(date +%H:%M:%S)  complete=$DONE running=$RUN pending=$MISS error=$ERR"
  [ $DONE -eq 9 ] && break
  if [ $ERR -gt 0 ] && [ $((DONE+ERR)) -eq 9 ]; then
    echo "ABORT: $ERR shard(s) failed and none are still running."; exit 1
  fi
  sleep $INTERVAL
done

echo
echo "=== downloading outputs ==="
for K in "${KERNELS[@]}"; do
  $KG kernels output "zeroxbhuvii/lassi2-$K" -p "$RAW" >/dev/null 2>&1 \
    && echo "  got $K" || { echo "  FAILED to download $K"; exit 1; }
done

# Shards emit matching_results_<Country>[_sNofM].tsv and the candidate twin.
find "$RAW" -name 'matching_results_*.tsv' -exec cp {} "$OUT"/ \;
find "$RAW" -name 'candidate_pairs_*.tsv'  -exec cp {} "$OUT"/ \;
echo "  partials: $(ls "$OUT"/matching_results_*.tsv 2>/dev/null | wc -l) matching, \
$(ls "$OUT"/candidate_pairs_*.tsv 2>/dev/null | wc -l) candidate"

echo
echo "=== merging ==="
LASSI_OUTPUT="$PWD/$OUT" PYTHONPATH="$SRC" .venv/bin/python "$SRC/pipeline.py" merge \
  --output-dir "$OUT" || exit 1

echo
echo "=== validating ==="
python3 student_resource/utils/validate_submission.py \
  --matching "$OUT/matching_results.tsv" \
  --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir student_resource/dataset/test --check-ids

echo
echo "rows: $(($(wc -l < "$OUT/matching_results.tsv") - 1))  (expected 1732544)"
echo "UPLOAD THIS FILE:  $PWD/$OUT/matching_results.tsv"
