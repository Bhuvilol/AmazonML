#!/bin/bash
# Launch the shards that chain_v2.sh could not.
#
# Kaggle caps concurrent batch CPU sessions at 5. chain_v2.sh pushed all nine in
# one burst: the five India shards took the slots, the remaining four were
# rejected with "Maximum batch CPU session count of 5 reached" -- and the chain
# printed "all shards launched" anyway, because it never checked the exit codes.
# This script retries each pending kernel until a slot frees, and verifies the
# push actually succeeded rather than trusting that it did.

set -uo pipefail
cd "$(dirname "$0")"
KG=.venv/bin/kaggle
PENDING=(us-s0of3 us-s1of3 us-s2of3 france)
INTERVAL=300          # 5 min between retries; shards run for hours

launched=()
failed=()

for K in "${PENDING[@]}"; do
  while :; do
    OUT=$($KG kernels push -p "kaggle/kernels_v2/$K" 2>&1)
    if grep -q "successfully pushed" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  LAUNCHED  $K"
      launched+=("$K")
      sleep 20                      # let Kaggle register the session
      break
    elif grep -q "Maximum batch CPU session count" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  waiting   $K (all 5 slots busy)"
      sleep $INTERVAL
    else
      echo "$(date +%H:%M:%S)  ERROR     $K: $OUT"
      failed+=("$K")
      break
    fi
  done
done

echo
echo "launched: ${#launched[@]}/${#PENDING[@]}  [${launched[*]:-none}]"
[ ${#failed[@]} -gt 0 ] && echo "FAILED:   ${failed[*]}" && exit 1
echo "all pending shards launched"
