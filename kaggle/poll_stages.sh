#!/bin/bash
# Poll Kaggle stages until genuinely complete, then download outputs.
#
# Distinguishes three states rather than two. An earlier version only grepped
# for RUNNING, so an auth failure looked identical to "finished" -- it reported
# success after 81 minutes when nothing had completed. Auth expires roughly
# every 3 hours and the CLI does not auto-refresh, so this case is common
# enough to handle explicitly rather than treat as done.
#
#   usage: poll_stages.sh india us france
set -u
KG="/Users/bhuvi/Documents/Projects/AmazonML/.venv/bin/kaggle"
STAGES=("$@")
START=$(date +%s)

while true; do
    PENDING=0
    for S in "${STAGES[@]}"; do
        OUT=$($KG kernels status "zeroxbhuvii/lassi-er-$S" 2>&1)
        if echo "$OUT" | grep -qi "Authentication required\|Permission .* denied"; then
            echo "AUTH EXPIRED after $(( ($(date +%s)-START)/60 )) min."
            echo "Kernels are still running on Kaggle -- only the API session died."
            echo "Re-run:  $KG auth login"
            exit 2
        fi
        echo "$OUT" | grep -q RUNNING && PENDING=$((PENDING+1))
    done
    [ $PENDING -eq 0 ] && break
    sleep 300
done

echo "=== all stages settled after $(( ($(date +%s)-START)/60 )) min ==="
for S in "${STAGES[@]}"; do
    printf "%-8s " "$S"; $KG kernels status "zeroxbhuvii/lassi-er-$S" 2>&1 | head -1
done

mkdir -p /tmp/kparts
for S in "${STAGES[@]}"; do
    mkdir -p "/tmp/kparts/$S"
    $KG kernels output "zeroxbhuvii/lassi-er-$S" -p "/tmp/kparts/$S" 2>&1 | tail -1
done
find /tmp/kparts -name "*.tsv" -exec ls -lh {} \; 2>/dev/null
