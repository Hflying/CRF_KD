#!/bin/bash
# R1-scale β-only KL ablation (matches r1_kdjp budget: 50 iso × 20 reads × 3 ep).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/phase2/r1_betaonly.out
OUT=checkpoints_kd/r1_betaonly

if [ -f "$OUT/eval.txt" ]; then
  echo "already done: $(grep overall_median "$OUT/eval.txt")"
  exit 0
fi

$PY -u scripts/train_r1_scale.py \
  --branch KD --preset m2_tiny_s5 \
  --reads_per_iso 20 --eval_reads_per_iso 20 \
  --epochs 3 --batch_size 16 --lr 2e-4 \
  --lambda_post 1.0 --lambda_hard 1.0 \
  --out_dir "$OUT" 2>&1 | tee "$LOG"

echo "DONE $(grep overall_median "$OUT/eval.txt")"
