#!/bin/bash
# Second-round bootstrap: use checkpoints_v2/best.pt (val_loss=0.78)
# to generate higher-quality CTC labels (wick_records_v3)
# Run this in a terminal with GPU access

set -e
cd /home/ubuntu/hardy/M2Caller

CKPT="checkpoints_v2/best.pt"
OUT_TRAIN="data/wick_records_v3/train"
OUT_VAL="data/wick_records_v3/val"
EXTRACTED="data/wick/train_extracted"

mkdir -p "$OUT_TRAIN" "$OUT_VAL"

echo "=== Second Bootstrap with $(basename $CKPT) ==="
echo "Output: $OUT_TRAIN"
echo ""

isolates=$(ls "$EXTRACTED")
total=$(echo "$isolates" | wc -l)
i=0

for name in $isolates; do
  i=$((i+1))
  isolate_dir="$EXTRACTED/$name"
  ref_fa="$isolate_dir/read_references.fasta"
  
  if [ ! -f "$ref_fa" ]; then
    echo "[$i/$total] SKIP $name (no ref)"
    continue
  fi
  
  n_fast5=$(ls "$isolate_dir/fast5/"*.fast5 2>/dev/null | wc -l)
  echo "[$i/$total] $name ($n_fast5 fast5 files)"
  
  conda run -n m2caller env PYTHONUNBUFFERED=1 python -u \
    -m m2caller.data.bootstrap_labels \
    --isolate_dir "$isolate_dir" \
    --ckpt        "$CKPT" \
    --out_dir     "$OUT_TRAIN" \
    --segment_len 4096 \
    --batch_size  64 \
    --shard_size  256 \
    --val_frac    0.05 \
    2>&1
    
  echo "[$i/$total] Done: $name"
done

echo ""
echo "=== Bootstrap v2 complete ==="
echo "Train shards: $(ls $OUT_TRAIN/*_signal.npy 2>/dev/null | wc -l)"
echo "Val   shards: $(ls $OUT_VAL/*_signal.npy 2>/dev/null | wc -l)"
