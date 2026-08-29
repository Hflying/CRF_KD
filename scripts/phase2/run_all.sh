#!/bin/bash
# Phase 2 master orchestrator (option B).
#
# Sequential run order (each stage waits for the previous):
#   1. run_main.sh                 — 4 main training runs (~470 GPU·h)
#   2. eval_nanospecies50.sh       — eval the 4 students on NanoSpecies-50
#   3. run_ablation.sh             — 5 ablation runs (~560 GPU·h)
#   4. eval_ablation50.sh          — eval the 5 ablation runs on NanoSpecies-50
#   5. throughput_benchmark.py     — reads/sec on the four students + sup/hac
#
# Each stage logs to logs/phase2/*.log and writes summary tables to
# logs/phase2/{main_table,ablation_table,throughput}.{md,tex,tsv,json}.
#
# To launch in the background and survive ssh disconnect:
#     nohup bash scripts/phase2/run_all.sh > logs/phase2/orch.out 2>&1 &
#     disown
#
# To check progress at any time:
#     tail -f logs/phase2/main.log     # currently-running stage
#     ls -lat checkpoints_kd/p2_*      # latest ckpt timestamps
#     bash scripts/phase2/status.sh    # summary
#
# To stop cleanly mid-run, kill the orchestrator pid; per-stage scripts use
# `set -e` so they will not start the next stage. Already-finished
# checkpoints under checkpoints_kd/p2_* are preserved.

set -e
cd "$(dirname "$0")/../.."
LOG_DIR=logs/phase2
mkdir -p "$LOG_DIR"
ORCH_LOG="$LOG_DIR/orch.log"
: > "$ORCH_LOG"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
section() { echo ""; echo "================================================="; echo "$1 @ $(stamp)"; echo "================================================="; }

{
section "Phase 2 orchestrator START"

section "Stage 1/5 — main training (run_main.sh)"
bash scripts/phase2/run_main.sh

section "Stage 2/5 — main eval on NanoSpecies-50 (eval_nanospecies50.sh)"
bash scripts/phase2/eval_nanospecies50.sh

section "Stage 3/5 — loss ablation (run_ablation.sh)"
bash scripts/phase2/run_ablation.sh

section "Stage 4/5 — ablation eval on NanoSpecies-50 (eval_ablation50.sh)"
bash scripts/phase2/eval_ablation50.sh

section "Stage 5/5 — throughput benchmark (throughput_benchmark.py)"
/home/ubuntu/miniconda3/envs/m2caller/bin/python -u scripts/phase2/throughput_benchmark.py \
    --out logs/phase2/throughput.json

section "Phase 2 orchestrator DONE"
echo "Final tables:"
ls -la logs/phase2/main_table.* logs/phase2/ablation_table.* logs/phase2/throughput.* 2>/dev/null

} 2>&1 | tee -a "$ORCH_LOG"
