#!/bin/bash
# Evaluate every Phase 2 student on NanoSpecies-50 × 100 reads/iso.
# Resulting CSVs match the schema of logs/bonito_sup_all.csv so they can be
# diffed directly. Each evaluation pass is ~2-3 GPU·h on a GH200.
#
# Run AFTER scripts/phase2/run_main.sh has produced the four checkpoints.

set -e
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval50.log"
: > "$LOG"

echo "=== NanoSpecies-50 student eval $(date) ===" | tee -a "$LOG"

declare -A RUNS=(
  [p2_h_tiny]="checkpoints_kd/p2_h_tiny/best.pt"
  [p2_kdjp_tiny]="checkpoints_kd/p2_kdjp_tiny/best.pt"
  [p2_h_small]="checkpoints_kd/p2_h_small/best.pt"
  [p2_kdjp_small]="checkpoints_kd/p2_kdjp_small/best.pt"
)

for tag in p2_h_tiny p2_kdjp_tiny p2_h_small p2_kdjp_small; do
  ckpt="${RUNS[$tag]}"
  out="$LOG_DIR/eval50_${tag}.csv"
  if [ ! -f "$ckpt" ]; then
      echo "[skip] $tag — checkpoint $ckpt not found" | tee -a "$LOG"
      continue
  fi
  if [ -f "$out" ]; then
      echo "[skip] $tag — $out already exists; remove to re-run" | tee -a "$LOG"
      continue
  fi
  echo "" | tee -a "$LOG"
  echo "=== eval $tag → $out ===" | tee -a "$LOG"
  $PY -u scripts/phase2/eval_student_nanospecies50.py \
      --ckpt "$ckpt" --out "$out" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== compare to Bonito sup baseline ===" | tee -a "$LOG"
$PY -u scripts/phase2/summarise_main.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== NanoSpecies-50 student eval done $(date) ===" | tee -a "$LOG"
