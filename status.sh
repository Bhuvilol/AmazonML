#!/bin/bash
# One-command view of everything running. Usage:  ./status.sh
KG="/Users/bhuvi/Documents/Projects/AmazonML/.venv/bin/kaggle"
TASKS="/private/tmp/claude-501/-Users-bhuvi-Documents-Projects-AmazonML/82ae76ab-2701-48c2-91a0-a20c2e62de54/tasks"
cd /Users/bhuvi/Documents/Projects/AmazonML

echo "════════════════════════════════════════════════════════════════"
echo "  LASSI — ER PIPELINE STATUS            $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"

echo
echo "── SCORES ──────────────────────────────────────────────────────"
echo "  leaderboard (submitted)  : 0.890575"
echo "  leaderboard TOP          : 0.985884     top-500 cutoff >0.95"
echo "  local validation         : 0.9099"
echo "  all-empty baseline       : 0.0582"

echo
echo "── LOCAL EXPERIMENTS ───────────────────────────────────────────"
# macOS ps has `etime` (D-HH:MM:SS), not `etimes`.
FOUND=0
for pat in recall_fix maxdf_real miss_analysis pr_split translit_test num_recall rebuild; do
    PID=$(pgrep -f "$pat" 2>/dev/null | head -1)
    if [ -n "$PID" ]; then
        ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
        printf "  ● RUNNING  %-18s  elapsed %s\n" "$pat" "${ELAPSED:-?}"
        FOUND=1
    fi
done
[ $FOUND -eq 0 ] && echo "  (none running)"

echo
echo "── LATEST EXPERIMENT OUTPUT ────────────────────────────────────"
# Exclude this script's own output, which would otherwise always be newest.
LATEST=$(grep -LE "LASSI — ER PIPELINE STATUS" "$TASKS"/*.output 2>/dev/null \
         | xargs ls -t 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    echo "  $(basename "$LATEST")"
    grep -viE "deprecat|polars.git|explode|^\s*$" "$LATEST" 2>/dev/null | tail -12 | sed 's/^/    /'
fi

echo
echo "── KAGGLE KERNELS ──────────────────────────────────────────────"
V2="lassi2-train lassi2-india-s0of5 lassi2-india-s1of5 lassi2-india-s2of5 \
    lassi2-india-s3of5 lassi2-india-s4of5 lassi2-us-s0of3 lassi2-us-s1of3 \
    lassi2-us-s2of3 lassi2-france"
DONE=0; RUN=0; ERR=0; PEND=0
for K in $V2; do
    OUT=$($KG kernels status "zeroxbhuvii/$K" 2>&1 | head -1)
    case "$OUT" in
        *RUNNING*)  printf "  ● %-22s RUNNING\n"  "$K"; RUN=$((RUN+1)) ;;
        *COMPLETE*) printf "  ✓ %-22s COMPLETE\n" "$K"; DONE=$((DONE+1)) ;;
        *ERROR*)    printf "  ✗ %-22s ERROR\n"    "$K"; ERR=$((ERR+1)) ;;
        *"Authentication required"*)
                    printf "  ? %-22s AUTH EXPIRED -> kaggle auth login --force\n" "$K" ;;
        *"wrong kernel slug"*|*denied*|*"not found"*|*404*)
                    printf "  · %-22s not launched yet\n" "$K"; PEND=$((PEND+1)) ;;
        *)          printf "  · %-22s %s\n" "$K" "$(echo "$OUT" | cut -c1-30)"; PEND=$((PEND+1)) ;;
    esac
done
echo "  ----------------------------------------"
printf "  %d complete · %d running · %d error · %d not launched\n" $DONE $RUN $ERR $PEND

echo
echo "── SUBMISSION FILES ────────────────────────────────────────────"
for f in Lassi_submission/output/matching_results.tsv Lassi_submission/output/candidate_pairs.tsv; do
    if [ -f "$f" ]; then
        printf "  %-42s %8s  %s rows\n" "$(basename "$f")" \
            "$(du -h "$f" | cut -f1)" "$(( $(wc -l < "$f") - 1 ))"
    fi
done

echo
echo "── REPO ────────────────────────────────────────────────────────"
printf "  HEAD %s   %s\n" "$(git rev-parse --short HEAD)" "$(git log -1 --format=%s | cut -c1-48)"
printf "  uncommitted: %s files\n" "$(git status --porcelain | wc -l | tr -d ' ')"

echo
echo "── WHERE TO WATCH ──────────────────────────────────────────────"
echo "  Kaggle live logs : https://www.kaggle.com/code/zeroxbhuvii/lassi-er-india"
echo "                     (open the kernel, then the 'Logs' / version tab)"
echo "  Local task logs  : $TASKS"
echo "  Follow one live  : tail -f \$(ls -t $TASKS/*.output | head -1)"
echo "════════════════════════════════════════════════════════════════"
