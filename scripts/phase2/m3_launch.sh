#!/bin/bash
# M3 — Phase 2 H-Small (Hard label, no KD).
# This is the missing baseline that ICASSP needs alongside p2_kdjp_small.
#
# Designed to coexist with other jobs on a shared GH200:
#   - default batch=16 → ~10-12 GB (vs 25 GB for batch=32)
#   - resumable: if killed (OOM, driver crash, ...), rerun the same
#     script and it will pick up from the last saved epoch{N}.pt
#   - epochs: default 10 (~3-4 days). Pass an arg to override:
#       bash m3_launch.sh                  # 10 ep, batch 16
#       bash m3_launch.sh 20               # 20 ep, batch 16
#       bash m3_launch.sh 20 32            # 20 ep, batch 32 (full GPU)
#
# Output: checkpoints_kd/p2_h_small/{epoch{N}.pt, best.pt, eval.{csv,txt}}.
# Log:    logs/phase2/m3.log + logs/phase2/m3.nohup.out

set -e
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOG_DIR=logs/phase2
CKPT_DIR=checkpoints_kd/p2_h_small
mkdir -p "$LOG_DIR" "$CKPT_DIR"
LOG="$LOG_DIR/m3.log"

EPOCHS=${1:-10}
BATCH=${2:-16}

# Safety: refuse to launch a duplicate M3 process on the same out_dir.
if pgrep -af "p2_h_small" > /dev/null; then
    echo "ABORT: another M3 (p2_h_small) process is already running."
    pgrep -af "p2_h_small"
    exit 1
fi

# Resume support: pick up from the highest existing epoch{N}.pt.
LAST_EP=$(ls -1 "$CKPT_DIR"/epoch*.pt 2>/dev/null \
    | sed -E 's/.*epoch([0-9]+)\.pt/\1/' | sort -n | tail -1)
INIT_ARG=""
START_EP=0
if [ -n "$LAST_EP" ]; then
    REMAIN=$((EPOCHS - LAST_EP))
    if [ "$REMAIN" -le 0 ]; then
        echo "Already at epoch $LAST_EP / $EPOCHS — nothing to do."
        echo "Delete $CKPT_DIR/epoch*.pt to retrain from scratch."
        exit 0
    fi
    echo "Resuming from epoch $LAST_EP, training $REMAIN more epoch(s)" | tee -a "$LOG"
    INIT_ARG="--init_ckpt $CKPT_DIR/epoch${LAST_EP}.pt --start_epoch $LAST_EP --no_warmup"
    START_EP=$LAST_EP
    EPOCHS=$REMAIN
fi

: >> "$LOG"
echo "" | tee -a "$LOG"
echo "=== M3 (P2 H-Small) start $(date) ===" | tee -a "$LOG"
echo "    epochs=$EPOCHS  batch=$BATCH  start_ep=$START_EP" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG" || { echo "GPU not available — aborting" | tee -a "$LOG"; exit 1; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== STAGE M3: H Small ${EPOCHS}ep batch=${BATCH} ===" | tee -a "$LOG"
$PY -u scripts/train_r1_scale.py --branch H \
    --preset m2_small_s5 \
    --reads_per_iso 100 --eval_reads_per_iso 20 \
    --epochs "$EPOCHS" --batch_size "$BATCH" --lr 2e-4 --warmup_steps 300 \
    $INIT_ARG \
    --out_dir "$CKPT_DIR" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== M3 done $(date) ===" | tee -a "$LOG"
[ -f $CKPT_DIR/eval.txt ] && \
    echo "FINAL: $(cat $CKPT_DIR/eval.txt | tr '\n' ' ')" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Next: bash scripts/phase2/eval_nanospecies50.sh" | tee -a "$LOG"
