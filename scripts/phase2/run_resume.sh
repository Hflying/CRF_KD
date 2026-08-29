#!/bin/bash
# Resume Phase 2 from where we paused on 2026-05-10.
#
# State as of 2026-05-10:
#   ✅ M1: H Tiny 20ep      → 85.61% (done)
#   ⏸  M2: KD Tiny stopped at epoch 8/20 → resume from epoch8.pt for 12 more epochs
#   ⏭  M3: SKIPPED (H Small from scratch)
#   ⏳ M4: KD Small +15ep    → not started; resume from r2_kdjp_small/epoch5.pt
#
# Total remaining wall-clock on a single GH200, batch 32:
#   M2 resume (12 ep): ~100h (since each epoch ≈ 8.3h)
#   M4 (15 ep):        ~160h
#   ----------------------------------------------
#   total              ~260h ≈ 11 days
set -e
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/resume.log"
: > "$LOG"

echo "=== Phase 2 RESUME start $(date) ===" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG" || { echo "GPU not available — aborting" | tee -a "$LOG"; exit 1; }

#-------------------------------------------------------------------------
# STAGE M2 (resume) — KD-Tiny from epoch8.pt for 12 more epochs (→ epoch 20)
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STAGE M2 (resume): KD Tiny ep8 → ep20 (+12 ep) ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_tiny_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 12 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --joint_post \
    --init_ckpt checkpoints_kd/p2_kdjp_tiny/epoch8.pt --start_epoch 8 \
    --out_dir checkpoints_kd/p2_kdjp_tiny 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# STAGE M4 — KD-Small, resume from R2 (+15 ep)
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STAGE M4: KD Small resume R2 +15ep ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 15 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --joint_post \
    --init_ckpt checkpoints_kd/r2_kdjp_small/epoch5.pt --start_epoch 5 \
    --out_dir checkpoints_kd/p2_kdjp_small 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 2 RESUME done $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== HEADLINE RESULTS ===" | tee -a "$LOG"
for d in p2_h_tiny p2_kdjp_tiny p2_kdjp_small; do
    f=checkpoints_kd/$d/eval.txt
    if [ -f "$f" ]; then
        printf "%-22s %s\n" "$d:" "$(tr '\n' ' ' < "$f")" | tee -a "$LOG"
    fi
done
