#!/bin/bash
# Evaluate every Phase 2 ablation student on NanoSpecies-50 × 100 reads/iso.
# ~12-15 GPU·h total (~3 h × 5).

set -e
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval50_ablation.log"
: > "$LOG"

echo "=== NanoSpecies-50 ablation eval $(date) ===" | tee -a "$LOG"

for tag in a1_hard a2_post a3_postfwd a4_filter a5_curric; do
    ckpt="checkpoints_kd/p2_abl_${tag}/best.pt"
    out="$LOG_DIR/eval50_abl_${tag}.csv"
    if [ ! -f "$ckpt" ]; then
        echo "[skip] $tag — checkpoint $ckpt not found" | tee -a "$LOG"
        continue
    fi
    if [ -f "$out" ]; then
        echo "[skip] $tag — $out already exists" | tee -a "$LOG"
        continue
    fi
    echo "" | tee -a "$LOG"
    echo "=== eval ablation $tag → $out ===" | tee -a "$LOG"
    $PY -u scripts/phase2/eval_student_nanospecies50.py \
        --ckpt "$ckpt" --out "$out" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== summarise ablation ===" | tee -a "$LOG"
$PY -u scripts/phase2/summarise_ablation.py 2>&1 | tee -a "$LOG"
