#!/bin/bash
# Architecture control: M2-Tiny with CNN backbone (not Mamba2) at R1 budget.
# Same teacher, same CRF head, same data as main runs — isolates backbone effect.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/phase2/arch_cnn.log
mkdir -p logs/phase2

run () {
  local tag=$1; shift
  local out=checkpoints_kd/arch_cnn_${tag}
  if [ -f "$out/eval.txt" ]; then
    echo "[skip] $tag $(grep overall_median "$out/eval.txt")" | tee -a "$LOG"
    return 0
  fi
  echo "[run] $tag $(date)" | tee -a "$LOG"
  $PY -u scripts/train_r1_scale.py "$@" \
    --preset m2_tiny_s5 --backbone simple \
    --reads_per_iso 20 --eval_reads_per_iso 20 \
    --epochs 3 --batch_size 16 --lr 2e-4 \
    --out_dir "$out" 2>&1 | tee -a "$LOG"
  echo "[done] $tag $(grep overall_median "$out/eval.txt")" | tee -a "$LOG"
}

echo "=== arch CNN baseline start $(date) ===" | tee "$LOG"

run h --branch H
run kd --branch KD --joint_post --lambda_post 1.0 --lambda_hard 1.0

echo "=== arch CNN summary ===" | tee -a "$LOG"
for tag in h kd; do
  f=checkpoints_kd/arch_cnn_${tag}/eval.txt
  [ -f "$f" ] && printf "  %-4s %s\n" "$tag" "$(tr '\n' ' ' < "$f")" | tee -a "$LOG"
done
echo "=== done $(date) ===" | tee -a "$LOG"
