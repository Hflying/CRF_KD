#!/bin/bash
# Wait for the shared GH200 to free up, then run the three BIBM revision
# experiments in order: (1) throughput, (2) lambda/seed sweep, (3) summarise.
#
# Usage:
#   nohup bash scripts/phase2/revexp_autolaunch.sh > logs/phase2/revexp_auto.out 2>&1 & disown
set -u
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOGDIR=logs/phase2
mkdir -p "$LOGDIR"
LOG="$LOGDIR/revexp_auto.log"

THRESHOLD_MIB=${1:-16000}
POLL_SEC=${2:-120}

echo "=== revexp autolaunch start $(date) (need >=${THRESHOLD_MIB} MiB) ===" | tee -a "$LOG"

while true; do
    if [ -f "$LOGDIR/throughput.csv" ] && [ -f "$LOG.sweep_done" ]; then
        echo "[$(date)] all experiments complete. exiting." | tee -a "$LOG"; exit 0
    fi
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$FREE" ]; then echo "[$(date)] no nvidia-smi; retry" | tee -a "$LOG"; sleep "$POLL_SEC"; continue; fi
    if [ "$FREE" -lt "$THRESHOLD_MIB" ]; then
        echo "[$(date)] free=${FREE} MiB < ${THRESHOLD_MIB} — waiting" | tee -a "$LOG"
        sleep "$POLL_SEC"; continue
    fi
    echo "[$(date)] free=${FREE} MiB — running experiments" | tee -a "$LOG"

    # (1) throughput
    if [ ! -f "$LOGDIR/throughput.csv" ]; then
        echo "[$(date)] >>> throughput" | tee -a "$LOG"
        $PY scripts/phase2/benchmark_throughput.py --n_reads 40 --batch_size 8 \
            --out "$LOGDIR/throughput.csv" > "$LOGDIR/throughput.out" 2>&1
        echo "[$(date)] throughput rc=$?" | tee -a "$LOG"
        tail -8 "$LOGDIR/throughput.out" | tee -a "$LOG"
    fi

    # (2) lambda/seed sweep (sequential, internal skip logic)
    echo "[$(date)] >>> lambda/seed sweep" | tee -a "$LOG"
    bash scripts/phase2/lambda_seed_sweep.sh >> "$LOGDIR/lambda_seed_sweep.out" 2>&1
    echo "[$(date)] sweep rc=$?" | tee -a "$LOG"
    touch "$LOG.sweep_done"

    # (3) summarise
    $PY scripts/phase2/summarise_sweep.py > "$LOGDIR/sweep_summary.txt" 2>&1
    cat "$LOGDIR/sweep_summary.txt" | tee -a "$LOG"
    echo "[$(date)] ALL DONE" | tee -a "$LOG"
    exit 0
done
