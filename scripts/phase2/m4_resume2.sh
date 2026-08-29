#!/bin/bash
# M4 resume after GPU driver crash on 2026-05-21 16:25 UTC.
# Continue from epoch12.pt for 8 more epochs (→ epoch20).
set -e
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/m4_resume2.log"
: > "$LOG"

echo "=== M4 RESUME-2 start $(date) ===" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG" || { echo "GPU not available — aborting" | tee -a "$LOG"; exit 1; }

echo "" | tee -a "$LOG"
echo "=== STAGE M4 (resume-2): KD Small ep12 → ep20 (+8 ep) ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 8 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --joint_post \
    --init_ckpt checkpoints_kd/p2_kdjp_small/epoch12.pt --start_epoch 12 \
    --out_dir checkpoints_kd/p2_kdjp_small 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== M4 RESUME-2 done $(date) ===" | tee -a "$LOG"
[ -f checkpoints_kd/p2_kdjp_small/eval.txt ] && \
    echo "FINAL: $(cat checkpoints_kd/p2_kdjp_small/eval.txt | tr '\n' ' ')" | tee -a "$LOG"
