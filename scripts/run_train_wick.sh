#!/bin/bash
# 直接用 wick_records (read_references.fasta 标签) 训练 M2Caller
# 学习率 3e-4，训 100 epoch
cd /home/ubuntu/hardy/M2Caller
mkdir -p checkpoints_wick2 logs

nohup conda run -n m2caller env PYTHONUNBUFFERED=1 python -u \
  -m m2caller.train \
  --train_records data/wick_records/train \
  --val_records   data/wick_records/val \
  --epochs        100 \
  --d_model       512 \
  --layers        2 \
  --ssm_d_state   128 \
  --ssm_d_conv    4 \
  --ssm_headdim   128 \
  --segment_len   4096 \
  --lr            3e-4 \
  --arch          mamba2 \
  --save_dir      checkpoints_wick2 \
  2>&1 | tee logs/train_wick2.log &

echo "M2Caller (wick_records, lr=3e-4, 100ep) PID: $!"
