#!/bin/bash
# Validation-scale CRF-KD experiments for the BIBM KD paper:
#   (A) lambda_post sensitivity sweep
#   (B) run-to-run seed repeats (H and KD)
#
# All runs: M2-Tiny, train on a single isolate (INF078), 200 reads, 4 epochs,
# joint posterior; eval on held-out reads of the same isolate. This matches
# the validation-scale ablation already in Table I, so results are comparable.
#
# Safe to nohup. Each run writes its own out_dir with eval.txt.
set -u
cd "$(dirname "$0")/../.."

PY="${PY:-python3}"
TRAIN=scripts/train_r1_scale.py
ISO=Klebsiella_pneumoniae_INF078
COMMON="--preset m2_tiny_s5 --train_iso_names $ISO \
  --reads_per_iso 200 --eval_reads_per_iso 50 --n_eval_iso 1 \
  --epochs 4 --batch_size 16 --lr 2e-4"
LOGDIR=logs/phase2
mkdir -p "$LOGDIR"
SWEEP_LOG="$LOGDIR/lambda_seed_sweep.log"
echo "=== lambda/seed sweep start $(date) ===" | tee -a "$SWEEP_LOG"

run () {  # name  extra_args
  local name="$1"; shift
  local out="checkpoints_kd/sweep_${name}"
  if [ -f "$out/eval.txt" ]; then
    echo "[skip] $name (eval.txt exists)" | tee -a "$SWEEP_LOG"
    return
  fi
  echo "[run] $name  $* $(date)" | tee -a "$SWEEP_LOG"
  $PY $TRAIN $COMMON --out_dir "$out" "$@" \
    > "$LOGDIR/sweep_${name}.out" 2>&1
  echo "[done] $name -> $(cat $out/eval.txt 2>/dev/null | tr '\n' ' ')" | tee -a "$SWEEP_LOG"
}

# (A) lambda_post sweep (KD, joint posterior, lambda_hard=1)
for LP in 0.25 0.5 1.0 2.0 4.0; do
  run "lam${LP}" --branch KD --joint_post --lambda_post "$LP" --lambda_hard 1.0 --seed 42
done

# (B) seed repeats: H (hard-only) and KD (lambda_post=1) x 3 seeds
for S in 42 1 7; do
  run "H_seed${S}"  --branch H  --seed "$S"
  run "KD_seed${S}" --branch KD --joint_post --lambda_post 1.0 --lambda_hard 1.0 --seed "$S"
done

echo "=== sweep all done $(date) ===" | tee -a "$SWEEP_LOG"
$PY scripts/phase2/summarise_sweep.py 2>&1 | tee -a "$SWEEP_LOG" || true
