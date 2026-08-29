#!/bin/bash
# Evaluate Bonito hac on all 50 NanoSpecies isolates.
# Output schema matches logs/bonito_sup_all.csv exactly.
# Wall-clock: ~6-12 GPU-hours on a GH200 (depends on chunking).

set -e
cd "$(dirname "$0")/../.."
PY=/home/ubuntu/miniconda3/envs/m2caller/bin/python
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bonito_hac_all.log"
OUT="$LOG_DIR/bonito_hac_all.csv"

if [ -f "$OUT" ]; then
    echo "ABORT: $OUT already exists. Remove it first to re-run."
    exit 1
fi

: > "$LOG"
echo "=== Bonito hac NanoSpecies-50 eval start $(date) ===" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

$PY -u scripts/phase2/eval_bonito_basecaller.py \
    --model tools/bonito_models/dna_r9.4.1_e8_hac@v3.3 \
    --out   "$OUT" \
    --n_reads 100 --batch_size 8 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Bonito hac done $(date) ===" | tee -a "$LOG"
