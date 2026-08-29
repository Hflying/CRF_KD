#!/bin/bash
# Phase 2 main runs (option B):
#   50 iso × 100 reads × 20 epochs.
#   Three branches RESUME from R2 epoch5 ckpts (+15 ep new training).
#   The 4th branch (H-Small) starts from scratch because R2 never trained it.
#
# Output checkpoints land in checkpoints_kd/p2_*/ ; logs in logs/phase2/.
# Each stage writes train.csv, eval.csv, eval.txt, and per-epoch ckpts.
#
# Per-stage wall-clock budget on a single GH200, extrapolated from R2:
#   STAGE M1  H-Tiny    (resume +15 ep)   ~64h
#   STAGE M2  KD-Tiny   (resume +15 ep)   ~125h
#   STAGE M3  H-Small   (from scratch 20) ~120h
#   STAGE M4  KD-Small  (resume +15 ep)   ~162h
#   ----------------------------------------------------
#   total                                  ~470h  (~20d)
#
# Resume strategy:
#   --init_ckpt loads the R2 epoch5 student weights only (no optimizer).
#   --start_epoch 5 means new ckpts are saved as epoch6.pt..epoch20.pt.
#   --no_warmup skips lr warmup (already past it on the source run).
#   --lr 1e-4 is half of R2's 2e-4 — a standard "fine-tune" choice
#     that should keep training stable without optimizer state.

set -e
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/main.log"
: > "$LOG"

echo "=== Phase 2 main start $(date) ===" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

#-------------------------------------------------------------------------
# STAGE M1 — H-Tiny, resume from R2 (+15 ep, ep5 → ep20)
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STAGE M1: H Tiny resume R2 +15ep ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch H \
    --preset m2_tiny_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 15 --batch_size 32 --lr 1e-4 --no_warmup \
    --init_ckpt checkpoints_kd/r2_h/epoch5.pt --start_epoch 5 \
    --out_dir checkpoints_kd/p2_h_tiny 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# STAGE M2 — KD-Tiny, resume from R2 (+15 ep)
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STAGE M2: KD Tiny resume R2 +15ep ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch KD \
    --preset m2_tiny_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs 15 --batch_size 32 --lr 1e-4 --no_warmup \
    --lambda_post 1.0 --lambda_hard 1.0 --joint_post \
    --init_ckpt checkpoints_kd/r2_kdjp_tiny/epoch5.pt --start_epoch 5 \
    --out_dir checkpoints_kd/p2_kdjp_tiny 2>&1 | tee -a "$LOG"

#-------------------------------------------------------------------------
# STAGE M3 — SKIPPED 2026-05-09. H-Small from scratch would cost ~120h but
# adds only a marginal ablation row; M4 (KD-Small) is the main paper number.
# The M3 block is preserved below (commented out) in case we want it later.
#-------------------------------------------------------------------------
echo "" | tee -a "$LOG"
echo "=== STAGE M3: SKIPPED (H Small from scratch — see comment in script) ===" | tee -a "$LOG"
# $PY -u scripts/train_r1_scale.py --branch H \
#     --preset m2_small_s5 \
#     --reads_per_iso 100 --eval_reads_per_iso 20 \
#     --epochs 20 --batch_size 32 --lr 2e-4 --warmup_steps 300 \
#     --out_dir checkpoints_kd/p2_h_small 2>&1 | tee -a "$LOG"

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
echo "=== Phase 2 main done $(date) ===" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== HEADLINE RESULTS (8-iso hold-out from training script's eval) ===" | tee -a "$LOG"
for d in p2_h_tiny p2_kdjp_tiny p2_h_small p2_kdjp_small; do
    f=checkpoints_kd/$d/eval.txt
    if [ -f "$f" ]; then
        printf "%-22s %s\n" "$d:" "$(tr '\n' ' ' < "$f")" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "Next: run scripts/phase2/eval_nanospecies50.sh to evaluate" | tee -a "$LOG"
echo "      all four students on the full NanoSpecies-50 × 100-read pool." | tee -a "$LOG"
