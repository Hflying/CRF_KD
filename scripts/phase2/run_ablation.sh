#!/bin/bash
# Phase 2 loss ablation:
#   M2_SMALL student × 10 epochs × 5 loss configurations.
#   Resume from R2-KD-Small epoch5.pt for the +post / +fwd / +filter / +curriculum
#   rows (saves ~50% wall-clock vs from-scratch).
#   The L_hard-only row resumes from the *hard* analogue if available, else
#   from R2-KD-Small (so all rows start from the same warm encoder).
#
# Per-stage budget on a single GH200, extrapolated from R2:
#   A1 hard only        ~108h  (Small, 10 ep, lr=1e-4, no warmup)
#   A2 + L_post         ~108h  (same, plus posterior KL)
#   A3 + L_post + L_fwd ~120h  (slight overhead from fwd-score MSE)
#   A4 + entropy filter ~110h  (mostly the same; filter cuts some chunks)
#   A5 + curriculum     ~115h  (curriculum scheduler, same compute)
#   ----------------------------------------------------------
#   total                ~560h  (~24 days single GH200)
#
# This is the slow half of Phase 2; if pressed for time, A4/A5 can be
# dropped to ~340h (14 days) with minor reviewer pushback.

set -e
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/ablation.log"
: > "$LOG"

# Prefer the hard-Small ckpt as init for A1 (matches its loss). If absent
# (run_main.sh hasn't produced p2_h_small yet), fall back to R2-KD-Small.
INIT_HARD=checkpoints_kd/p2_h_small/epoch5.pt
[ -f "$INIT_HARD" ] || INIT_HARD=checkpoints_kd/r2_kdjp_small/epoch5.pt

INIT_KD=checkpoints_kd/r2_kdjp_small/epoch5.pt

echo "=== Phase 2 ablation start $(date) ===" | tee -a "$LOG"
echo "INIT_HARD = $INIT_HARD" | tee -a "$LOG"
echo "INIT_KD   = $INIT_KD"   | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

#-------------------------------------------------------------------------
# A1 — L_hard only
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== A1: L_hard only ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch H \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 10 --batch_size 32 --lr 1e-4 --no_warmup \
    --init_ckpt "$INIT_HARD" --start_epoch 5 \
    --out_dir checkpoints_kd/p2_abl_a1_hard 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# A2 — + L_post
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== A2: + L_post ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 10 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --lambda_fwd 0.0 --joint_post \
    --init_ckpt "$INIT_KD" --start_epoch 5 \
    --out_dir checkpoints_kd/p2_abl_a2_post 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# A3 — + L_post + L_fwd
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== A3: + L_post + L_fwd ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 10 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --lambda_fwd 0.5 --joint_post \
    --init_ckpt "$INIT_KD" --start_epoch 5 \
    --out_dir checkpoints_kd/p2_abl_a3_postfwd 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# A4 — + entropy filter (no curriculum; static 80th percentile)
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== A4: + entropy filter ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 10 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --lambda_fwd 0.5 --joint_post \
    --use_entropy_filter --entropy_pct 80 \
    --init_ckpt "$INIT_KD" --start_epoch 5 \
    --out_dir checkpoints_kd/p2_abl_a4_filter 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# A5 — + curriculum (50→100 percentile over 6 epochs out of 10)
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== A5: + curriculum ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 10 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --lambda_fwd 0.5 --joint_post \
    --use_entropy_filter --use_curriculum \
    --curriculum_warmup_epochs 6 --curriculum_start_pct 50 \
    --init_ckpt "$INIT_KD" --start_epoch 5 \
    --out_dir checkpoints_kd/p2_abl_a5_curric 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Phase 2 ablation done $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== ABLATION HEADLINE (8-iso hold-out) ===" | tee -a "$LOG"
for d in p2_abl_a1_hard p2_abl_a2_post p2_abl_a3_postfwd p2_abl_a4_filter p2_abl_a5_curric; do
    f=checkpoints_kd/$d/eval.txt
    if [ -f "$f" ]; then
        printf "%-20s %s\n" "$d:" "$(tr '\n' ' ' < "$f")" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Next: run scripts/phase2/eval_ablation50.sh for the NanoSpecies-50 eval." | tee -a "$LOG"
