#!/bin/bash
# Launch v3 (competition features, +0.0094 measured) as soon as Kaggle frees a
# slot, then its 9 predict shards as the v2 shards finish.
#
# All kernels are pinned to commit c6348a4 -- the commit whose code trains the
# 32-feature model they load. An unpinned kernel clones whatever is on main when
# it STARTS, which is exactly what killed the v2 France shard after 113 minutes.
set -uo pipefail
cd "$(dirname "$0")"
KG=.venv/bin/kaggle
INTERVAL=300

# Default to RETRYING, not to giving up.
#
# Three times now a transient condition has been treated as fatal and killed an
# unattended run: unchecked shard pushes, then auth expiry, then a DNS blip
# ("Failed to resolve api.kaggle.com") that ended a launcher which had already
# waited 2h15m for a slot. Whitelisting each new transient error as it appears
# is a losing game -- the correct default for a long-running launcher is to keep
# trying and only give up after many CONSECUTIVE unknown failures.
MAX_UNKNOWN=20        # ~100 min of consecutive unrecognised errors before quitting

push_until_free() {   # $1 = kernel dir name
  local unknown=0
  while :; do
    OUT=$($KG kernels push -p "kaggle/kernels_v3/$1" 2>&1)
    if grep -q "successfully pushed" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  LAUNCHED  lassi3-$1"; sleep 20; return 0
    elif grep -q "Maximum batch CPU session count" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  waiting   lassi3-$1 (slots full)"; unknown=0
    elif grep -qi "Authentication required\|denied\|401\|403" <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  AUTH EXPIRED -- run: kaggle auth login --force"; unknown=0
    elif grep -qiE "Max retries|NameResolution|Connection|timed out|TLS|SSL|50[0-9] " <<<"$OUT"; then
      echo "$(date +%H:%M:%S)  network blip, retrying lassi3-$1"; unknown=0
    else
      unknown=$((unknown+1))
      echo "$(date +%H:%M:%S)  unknown error $unknown/$MAX_UNKNOWN lassi3-$1: $(head -1 <<<"$OUT")"
      [ $unknown -ge $MAX_UNKNOWN ] && { echo "  giving up on $1"; return 1; }
    fi
    sleep $INTERVAL
  done
}

echo "=== v3 training ==="
push_until_free train || exit 1

echo
echo "=== waiting for v3 training to finish ==="
while :; do
  S=$($KG kernels status zeroxbhuvii/lassi3-train 2>&1 | tail -1)
  case "$S" in
    *COMPLETE*) echo "$(date +%H:%M:%S)  training COMPLETE"; break ;;
    *ERROR*|*CANCEL*) echo "$(date +%H:%M:%S)  TRAINING FAILED"; exit 1 ;;
    # Anything else -- running, queued, auth gone, DNS gone -- means keep waiting.
    *) echo "$(date +%H:%M:%S)  training not finished yet" ;;
  esac
  sleep $INTERVAL
done

echo
echo "=== publishing v3 model as a dataset version ==="
rm -rf /tmp/v3ds && mkdir -p /tmp/v3ds
$KG kernels output zeroxbhuvii/lassi3-train -p /tmp/v3out >/dev/null 2>&1
find /tmp/v3out -name 'model.txt' -o -name 'threshold.json' -o -name 'blocking_recall.json' \
  | while read -r f; do cp "$f" /tmp/v3ds/; done
cp kaggle/kernels_v2/train/../../dataset-metadata.json /tmp/v3ds/ 2>/dev/null || \
  $KG datasets init -p /tmp/v3ds >/dev/null 2>&1
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/tmp/v3ds/dataset-metadata.json")
m = json.loads(p.read_text())
m["id"] = "zeroxbhuvii/lassi-er-artifacts"
m["title"] = "lassi-er-artifacts"
p.write_text(json.dumps(m, indent=2))
PY
$KG datasets version -p /tmp/v3ds -m "v3: competition features (32)" --dir-mode zip 2>&1 | tail -2
sleep 60

echo
echo "=== launching 9 v3 predict shards ==="
for K in india-s0of5 india-s1of5 india-s2of5 india-s3of5 india-s4of5 \
         us-s0of3 us-s1of3 us-s2of3 france; do
  push_until_free "$K" || echo "  gave up on $K"
done
echo
echo "all v3 shards launched at $(date '+%H:%M')"
