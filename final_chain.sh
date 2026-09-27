#!/bin/bash
# Production chain: wait for training -> publish model -> launch 9 shards.
# Retries transient errors by default (auth expiry, DNS, slot cap) because
# every previous unattended run was killed by treating one as fatal.
set -uo pipefail
cd "$(dirname "$0")"
KG=.venv/bin/kaggle
INT=180
MAXUNK=20

echo "=== waiting for lassifinal-train ==="
while :; do
  S=$($KG kernels status zeroxbhuvii/lassifinal-train 2>&1 | tail -1)
  case "$S" in
    *COMPLETE*) echo "$(date +%H:%M:%S)  TRAINING COMPLETE"; break ;;
    *ERROR*|*CANCEL*) echo "$(date +%H:%M:%S)  TRAINING FAILED"; exit 1 ;;
    *) echo "$(date +%H:%M:%S)  training running" ;;
  esac
  sleep $INT
done

echo; echo "=== publishing model as dataset version ==="
rm -rf /tmp/finalds /tmp/finalout && mkdir -p /tmp/finalds
$KG kernels output zeroxbhuvii/lassifinal-train -p /tmp/finalout >/dev/null 2>&1
find /tmp/finalout \( -name model.txt -o -name threshold.json -o -name blocking_recall.json \) \
  -exec cp {} /tmp/finalds/ \;
ls -la /tmp/finalds/
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/tmp/finalds/dataset-metadata.json")
p.write_text(json.dumps({"id": "zeroxbhuvii/lassi-er-artifacts",
                         "title": "lassi-er-artifacts"}, indent=2))
PY
$KG datasets version -p /tmp/finalds -m "FINAL 43-feature production model (18134aa)" --dir-mode zip 2>&1 | tail -2
sleep 90

echo; echo "=== launching 9 predict shards ==="
push_one() {
  local unk=0
  while :; do
    OUT=$($KG kernels push -p "kaggle/kernels_final/$1" 2>&1)
    if grep -q "successfully pushed" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  LAUNCHED  $1"; sleep 15; return 0
    elif grep -q "Maximum batch CPU session count" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  waiting   $1 (slots full)"; unk=0
    elif grep -qiE "Authentication|denied|401|403" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  AUTH EXPIRED -> kaggle auth login --force"; unk=0
    elif grep -qiE "Max retries|NameResolution|Connection|timed out|SSL|50[0-9] " <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  network blip, retrying $1"; unk=0
    else
      unk=$((unk+1)); echo "$(date +%H:%M:%S)  unknown $unk/$MAXUNK $1: $(head -1 <<<"$OUT")"
      [ $unk -ge $MAXUNK ] && return 1
    fi
    sleep $INT
  done
}
for K in india-s0of5 india-s1of5 india-s2of5 india-s3of5 india-s4of5 \
         us-s0of3 us-s1of3 us-s2of3 france; do
  push_one "$K" || echo "  GAVE UP on $K"
done
echo; echo "all shards launched at $(date '+%H:%M')"
