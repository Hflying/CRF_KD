#!/bin/bash
# Wait for GPU compute to be idle, then run R1 β-only ablation.
set -u
cd "$(dirname "$0")/../.."
LOG=logs/phase2/r1_betaonly_auto.log
UTIL_MAX=${1:-20}
POLL=${2:-180}

echo "=== r1_betaonly autolaunch $(date) util<=${UTIL_MAX}% ===" | tee -a "$LOG"

while true; do
  if [ -f checkpoints_kd/r1_betaonly/eval.txt ]; then
    echo "[$(date)] done: $(grep overall_median checkpoints_kd/r1_betaonly/eval.txt)" | tee -a "$LOG"
    echo "R1_BETAONLY_DONE"
    exit 0
  fi
  U1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  sleep 10
  U2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${U1:-100}" -le "$UTIL_MAX" ] && [ "${U2:-100}" -le "$UTIL_MAX" ]; then
    echo "[$(date)] util=${U1}%/${U2}% — launching" | tee -a "$LOG"
    bash scripts/phase2/run_r1_betaonly.sh >> "$LOG" 2>&1 || true
  else
    echo "[$(date)] util=${U1}%/${U2}% — waiting" | tee -a "$LOG"
  fi
  sleep "$POLL"
done
