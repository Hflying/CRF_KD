#!/bin/bash
# One-line summary of where Phase 2 is at.
cd "$(dirname "$0")/../.."

echo "==== Phase 2 status @ $(date '+%Y-%m-%d %H:%M:%S') ===="
echo ""
printf "%-22s %-8s %-12s %s\n" "RUN" "STATE" "LATEST_EP" "MEDIAN_8ISO"
printf "%-22s %-8s %-12s %s\n" "----------------------" "--------" "------------" "------------"

for d in p2_h_tiny p2_kdjp_tiny p2_h_small p2_kdjp_small \
         p2_abl_a1_hard p2_abl_a2_post p2_abl_a3_postfwd \
         p2_abl_a4_filter p2_abl_a5_curric; do
    base="checkpoints_kd/$d"
    if [ ! -d "$base" ]; then
        printf "%-22s %-8s %-12s %s\n" "$d" "—" "—" "—"; continue
    fi
    ep=$(ls -1 "$base"/epoch*.pt 2>/dev/null | sed -E 's/.*epoch([0-9]+)\.pt/\1/' | sort -n | tail -1)
    [ -z "$ep" ] && ep="—"
    state="running"
    [ -f "$base/eval.txt" ] && state="done"
    med="—"
    [ -f "$base/eval.txt" ] && med=$(grep overall_median "$base/eval.txt" | awk '{print $2}')
    printf "%-22s %-8s %-12s %s\n" "$d" "$state" "$ep" "$med"
done

echo ""
echo "Latest activity:"
ls -lat --time-style='+%Y-%m-%d %H:%M' logs/phase2/*.log 2>/dev/null | head -5
echo ""
echo "Completed eval CSVs:"
ls -1 logs/phase2/eval50_*.csv 2>/dev/null
echo ""
echo "Summary tables:"
ls -1 logs/phase2/{main_table,ablation_table,throughput}.* 2>/dev/null
