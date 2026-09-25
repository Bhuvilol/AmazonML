#!/bin/bash
# Wait for v2 training, republish the model as a dataset, launch all 9 shards.
#
# The model must go through a DATASET, not kernel_sources -- kernel chaining
# silently fails to mount on Kaggle (cost three failed runs earlier).
set -u
KG="/Users/bhuvi/Documents/Projects/AmazonML/.venv/bin/kaggle"
cd /Users/bhuvi/Documents/Projects/AmazonML

echo "waiting for lassi2-train ..."
while $KG kernels status zeroxbhuvii/lassi2-train 2>&1 | grep -q RUNNING; do sleep 300; done
STATUS=$($KG kernels status zeroxbhuvii/lassi2-train 2>&1 | head -1)
echo "$STATUS"
echo "$STATUS" | grep -q COMPLETE || { echo "TRAIN DID NOT COMPLETE — stopping"; exit 1; }

rm -rf /tmp/v2train && mkdir -p /tmp/v2train
$KG kernels output zeroxbhuvii/lassi2-train -p /tmp/v2train 2>&1 | tail -1
./.venv/bin/python -c "
import json,glob
for f in glob.glob('/tmp/v2train/*.log'):
    try:
        for e in json.load(open(f)):
            s=e.get('data','') if isinstance(e,dict) else str(e)
            if s.strip(): print(s.rstrip())
    except Exception: pass
" 2>/dev/null | grep -E "BEST THRESHOLD|recall ceiling|exact-key|transliterated" | head -8

[ -f /tmp/v2train/artifacts/model.txt ] || { echo "NO MODEL PRODUCED — stopping"; exit 1; }

echo; echo "publishing new model as dataset version ..."
rm -rf /tmp/v2ds && mkdir -p /tmp/v2ds
cp /tmp/v2train/artifacts/*.json /tmp/v2train/artifacts/model.txt /tmp/v2ds/
cat > /tmp/v2ds/dataset-metadata.json <<'META'
{"title":"lassi-er-artifacts","id":"zeroxbhuvii/lassi-er-artifacts","licenses":[{"name":"CC0-1.0"}]}
META
$KG datasets version -p /tmp/v2ds -m "v2: max_df=0.5, exact-key, transliteration" --dir-mode zip 2>&1 | tail -2
sleep 60

echo; echo "launching 9 predict shards ..."
for K in india-s0of5 india-s1of5 india-s2of5 india-s3of5 india-s4of5 \
         us-s0of3 us-s1of3 us-s2of3 france; do
    $KG kernels push -p kaggle/kernels_v2/$K 2>&1 | head -1
    sleep 5
done
echo; echo "all shards launched at $(date '+%H:%M')"
