#!/bin/bash
# Wait for the shared GH200 to free up, then run ONLY the lambda sweep
# (the seed runs already have eval.txt and are skipped). Lambda runs are
# small (M2-Tiny + teacher, single isolate, batch 16) so ~12 GB suffices.
set -u
cd "$(dirname "$0")/../.."
LOGDIR=logs/phase2
mkdir -p "$LOGDIR"
LOG="$LOGDIR/lambda_auto.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

THRESHOLD_MIB=${1:-12000}
POLL_SEC=${2:-120}
UTIL_MAX=${3:-20}
echo "=== lambda autolaunch start $(date) (need >=${THRESHOLD_MIB} MiB) ===" | tee -a "$LOG"

all_done () {
    for lp in 0.25 0.5 1.0 2.0 4.0; do
        [ -f "checkpoints_kd/sweep_lam${lp}/eval.txt" ] || return 1
    done
    return 0
}

while true; do
    if all_done; then
        echo "[$(date)] all 5 lambda runs have eval.txt — done" | tee -a "$LOG"
        "${PY:-python3}" scripts/phase2/summarise_sweep.py \
            > "$LOGDIR/sweep_summary.txt" 2>&1
        echo "LAMBDA_ALL_DONE" | tee -a "$LOG"
        cat "$LOGDIR/sweep_summary.txt" | tee -a "$LOG"
        exit 0
    fi
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$FREE" ]; then sleep "$POLL_SEC"; continue; fi
    if [ "$FREE" -lt "$THRESHOLD_MIB" ]; then
        echo "[$(date)] free=${FREE} MiB < ${THRESHOLD_MIB} — waiting" | tee -a "$LOG"
        sleep "$POLL_SEC"; continue
    fi
    # memory-free is not enough: other users can saturate COMPUTE while
    # using little memory. Require the GPU to be genuinely idle (low util)
    # across two samples ~10s apart, else our run crawls at ~30 s/step.
    U1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    sleep 10
    U2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "${U1:-100}" -gt "$UTIL_MAX" ] || [ "${U2:-100}" -gt "$UTIL_MAX" ]; then
        echo "[$(date)] free=${FREE} MiB but util=${U1}%/${U2}% > ${UTIL_MAX}% — compute busy, waiting" | tee -a "$LOG"
        sleep "$POLL_SEC"; continue
    fi
    echo "[$(date)] free=${FREE} MiB, util=${U1}%/${U2}% — GPU idle, running lambda sweep" | tee -a "$LOG"
    bash scripts/phase2/lambda_seed_sweep.sh >> "$LOGDIR/lambda_rerun.out" 2>&1
    echo "[$(date)] sweep pass rc=$? (will re-check completeness)" | tee -a "$LOG"
    sleep 5
done
