#!/bin/bash
# Evaluate Dorado sup v3.6 on all 50 NanoSpecies isolates via PyTorch loader.
# Uses the same audited Viterbi decoder as Bonito sup/hac evals.
set -e
cd "$(dirname "$0")/../.."
PY="${PY:-python3}"
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/dorado_sup_all.log"
OUT="$LOG_DIR/dorado_sup_all.csv"
MODEL=tools/dorado_models/dna_r9.4.1_e8_sup@v3.6

if [ -f "$OUT" ]; then
    echo "ABORT: $OUT already exists. Remove it first to re-run."
    exit 1
fi

: > "$LOG"
echo "=== Dorado sup v3.6 NanoSpecies-50 eval start $(date) ===" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

$PY -u scripts/phase2/eval_bonito_basecaller.py \
    --dorado \
    --model "$MODEL" \
    --out   "$OUT" \
    --n_reads 100 --batch_size 8 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Dorado sup done $(date) ===" | tee -a "$LOG"
