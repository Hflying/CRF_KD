#!/bin/bash
# Wait until the shared GH200 has enough free memory, then launch M3.
# Polls every 2 minutes; starts M3 (resume from latest epoch) once
# free memory >= THRESHOLD_MIB. Safe to nohup-and-forget.
#
# Usage:
#   nohup bash scripts/phase2/m3_autolaunch.sh > logs/phase2/m3_auto.nohup.out 2>&1 & disown

set -u
cd "$(dirname "$0")/../.."
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/m3_auto.log"

THRESHOLD_MIB=${1:-14000}   # need ~10-12 GB for batch=16; 14 GB gives headroom
EPOCHS=${2:-20}
POLL_SEC=${3:-120}

echo "=== M3 autolaunch watcher start $(date) ===" | tee -a "$LOG"
echo "    waiting for >= ${THRESHOLD_MIB} MiB free, poll every ${POLL_SEC}s" | tee -a "$LOG"

while true; do
    # Already done?
    if [ -f checkpoints_kd/p2_h_small/eval.txt ]; then
        echo "[$(date)] eval.txt exists — M3 already complete. Exiting watcher." | tee -a "$LOG"
        exit 0
    fi
    # Already running?
    if pgrep -af "train_r1_scale.*p2_h_small" > /dev/null; then
        echo "[$(date)] M3 already running. Exiting watcher." | tee -a "$LOG"
        exit 0
    fi
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$FREE" ]; then
        echo "[$(date)] nvidia-smi unavailable; retrying" | tee -a "$LOG"
        sleep "$POLL_SEC"; continue
    fi
    if [ "$FREE" -ge "$THRESHOLD_MIB" ]; then
        echo "[$(date)] free=${FREE} MiB >= ${THRESHOLD_MIB} — launching M3" | tee -a "$LOG"
        nohup bash scripts/phase2/m3_launch.sh "$EPOCHS" \
            > "$LOG_DIR/m3_resume.nohup.out" 2>&1 &
        echo $! | tee "$LOG_DIR/m3_resume.pid"
        echo "[$(date)] launched pid $(cat $LOG_DIR/m3_resume.pid)" | tee -a "$LOG"
        sleep 30
        tail -5 "$LOG_DIR/m3.log" | tee -a "$LOG"
        exit 0
    else
        echo "[$(date)] free=${FREE} MiB < ${THRESHOLD_MIB} — waiting" | tee -a "$LOG"
    fi
    sleep "$POLL_SEC"
done
