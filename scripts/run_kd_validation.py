"""
Small-scale CRF-KD validation: trains an ``m2_tiny_s5`` CRFStudent on a
subset of pre-existing wick_records_v3 shards. Two modes:

  --mode hard : Hard-label-only baseline (lambda_post=0, lambda_fwd=0,
                lambda_hard=1) -- no teacher forward needed, uses Bonito
                movetable labels already in the shards.
  --mode kd   : Full CRF-KD (lambda_post=1, lambda_fwd=0.5, lambda_hard=1)
                -- runs a frozen Bonito sup teacher forward each step.

Both modes share student arch, optimiser, schedule, seed, and data, so the
relative read-level identity gap on a held-out set is attributable to the
soft KD losses.

Usage:
    python scripts/run_kd_validation.py \
        --mode kd --epochs 8 --batch_size 8 \
        --train_shards 6 --val_shards 2 \
        --out_dir checkpoints_kd/validation/KD
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import List, Tuple

sys.path.insert(0, "/home/ubuntu/miniconda3/envs/m2caller/lib/python3.10/site-packages")
sys.path.insert(0, "/home/ubuntu/hardy/M2Caller")
sys.path.insert(0, "/tmp/edpkg")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m2caller.kd.crf_kd import CRFKDLoss
from m2caller.kd.student_configs import build_student
from m2caller.utils.labeled_dataset import LabeledNumpyRecordsDataset


def list_shard_pairs(records_dir: str, max_shards: int) -> List[str]:
    sig_files = sorted(
        f for f in os.listdir(records_dir) if f.endswith("_signal.npy")
    )
    pairs = []
    for s in sig_files:
        lbl = s.replace("_signal.npy", "_labels.npz")
        if os.path.exists(os.path.join(records_dir, lbl)):
            pairs.append(s)
        if len(pairs) >= max_shards:
            break
    return pairs


class _ShardSubset(LabeledNumpyRecordsDataset):
    """LabeledNumpyRecordsDataset restricted to a fixed list of shards."""

    def __init__(self, records_dir: str, shards: List[str]):
        self.records_dir = records_dir
        self.half = False
        self.shards = shards


def shard_batch_iter(sig: torch.Tensor,
                     seqs: torch.Tensor,
                     lens: torch.Tensor,
                     batch_size: int):
    """Iterate over (batch_size,) chunks from one shard."""
    n = sig.shape[0]
    for i in range(0, n, batch_size):
        yield (sig[i:i + batch_size],
               seqs[i:i + batch_size],
               lens[i:i + batch_size])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hard", "kd"], required=True)
    ap.add_argument("--train_dir", default="data/wick_records_v3/train")
    ap.add_argument("--val_dir", default="data/wick_records_v3/val")
    ap.add_argument("--train_shards", type=int, default=6,
                    help="Number of shards to sample for training "
                         "(each shard ~256 chunks of 4096 samples).")
    ap.add_argument("--val_shards", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--student_preset", default="m2_tiny_s5")
    ap.add_argument("--teacher_model",
                    default="tools/bonito_models/dna_r9.4.1_e8_sup@v3.3")
    ap.add_argument("--lambda_post", type=float, default=1.0)
    ap.add_argument("--lambda_fwd", type=float, default=0.5)
    ap.add_argument("--lambda_hard", type=float, default=1.0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if args.mode == "hard":
        args.lambda_post = 0.0
        args.lambda_fwd = 0.0
        args.lambda_hard = 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda")
    print(f"[{args.mode}] device={device}", flush=True)

    # ---- Data ----
    train_shards = list_shard_pairs(args.train_dir, args.train_shards)
    val_shards = list_shard_pairs(args.val_dir, args.val_shards)
    print(f"[{args.mode}] train_shards={len(train_shards)}  "
          f"val_shards={len(val_shards)}", flush=True)
    train_ds = _ShardSubset(args.train_dir, train_shards)
    val_ds = _ShardSubset(args.val_dir, val_shards)

    # ---- Student ----
    student = build_student(args.student_preset).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[{args.mode}] student_preset={args.student_preset}  "
          f"params={n_params/1e6:.2f}M  state_len={student.state_len}",
          flush=True)

    # ---- Teacher (KD only) ----
    teacher = None
    if args.mode == "kd":
        from bonito.util import load_model
        print(f"[{args.mode}] loading teacher {args.teacher_model}", flush=True)
        teacher = load_model(args.teacher_model, device="cuda")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        if teacher.seqdist.state_len != student.seqdist.state_len:
            warnings.warn(
                f"state_len mismatch: teacher={teacher.seqdist.state_len} "
                f"vs student={student.seqdist.state_len}",
                RuntimeWarning,
            )

    # ---- Loss / Optim ----
    kd_loss = CRFKDLoss(
        lambda_post=args.lambda_post,
        lambda_fwd=args.lambda_fwd,
        lambda_hard=args.lambda_hard,
    ).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    best_val = float("inf")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        student.train()
        tot_loss = tot_post = tot_fwd = tot_hard = 0.0
        n_steps = 0
        for shard_idx in range(len(train_ds)):
            _, sig_np, seqs_np, lens_np = train_ds[shard_idx]
            sig = torch.from_numpy(sig_np).squeeze(-1)         # (N, W)
            seqs = torch.from_numpy(seqs_np).long()            # (N, L)
            lens = torch.from_numpy(lens_np).int()             # (N,)
            for sb, eb, lb in shard_batch_iter(sig, seqs, lens, args.batch_size):
                sb = sb.to(device).unsqueeze(1)                # (B, 1, W)
                eb = eb.to(device)
                lb = lb.to(device)

                if args.mode == "kd":
                    with torch.no_grad():
                        teacher_scores = teacher(sb.half()).float()
                else:
                    teacher_scores = torch.zeros(1, device=device)

                student_scores = student(sb).float()

                if args.mode == "kd":
                    losses = kd_loss(
                        teacher_scores=teacher_scores,
                        student_scores=student_scores,
                        teacher_targets=eb,
                        target_lengths=lb,
                        teacher_seqdist=teacher.seqdist,
                        student_seqdist=student.seqdist,
                    )
                else:
                    losses = kd_loss(
                        teacher_scores=student_scores,  # unused (lambdas=0)
                        student_scores=student_scores,
                        teacher_targets=eb,
                        target_lengths=lb,
                        teacher_seqdist=student.seqdist,
                        student_seqdist=student.seqdist,
                    )
                loss = losses["total"]

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()

                tot_loss += float(loss)
                tot_post += float(losses["post"])
                tot_fwd += float(losses["fwd"])
                tot_hard += float(losses["hard"])
                n_steps += 1

        avg = tot_loss / max(1, n_steps)
        avg_post = tot_post / max(1, n_steps)
        avg_fwd = tot_fwd / max(1, n_steps)
        avg_hard = tot_hard / max(1, n_steps)

        # ---- Validate ----
        student.eval()
        v_loss = 0.0
        v_steps = 0
        with torch.no_grad():
            for shard_idx in range(len(val_ds)):
                _, sig_np, seqs_np, lens_np = val_ds[shard_idx]
                sig = torch.from_numpy(sig_np).squeeze(-1)
                seqs = torch.from_numpy(seqs_np).long()
                lens = torch.from_numpy(lens_np).int()
                for sb, eb, lb in shard_batch_iter(sig, seqs, lens, args.batch_size):
                    sb = sb.to(device).unsqueeze(1)
                    eb = eb.to(device)
                    lb = lb.to(device)

                    student_scores = student(sb).float()
                    # Use only L_hard for val to keep eval cheap and
                    # comparable across modes.
                    from m2caller.kd.crf_kd import l_hard_kd
                    val_l = l_hard_kd(student_scores, eb, lb, student.seqdist)
                    v_loss += float(val_l)
                    v_steps += 1
        avg_v = v_loss / max(1, v_steps)
        elapsed = time.time() - t_start
        print(f"[{args.mode}] epoch={epoch}/{args.epochs} steps={n_steps} "
              f"train_total={avg:.4f} hard={avg_hard:.4f} "
              f"post={avg_post:.4f} fwd={avg_fwd:.4f}  "
              f"val_hard={avg_v:.4f}  elapsed={elapsed:.0f}s",
              flush=True)

        if avg_v < best_val:
            best_val = avg_v
            ckpt_path = os.path.join(args.out_dir, "best.pt")
            torch.save({
                "student": student.state_dict(),
                "preset": args.student_preset,
                "epoch": epoch,
                "val_hard": avg_v,
                "train_total": avg,
                "args": vars(args),
            }, ckpt_path)
            print(f"[{args.mode}]  -> saved {ckpt_path} (val_hard={avg_v:.4f})",
                  flush=True)

    print(f"[{args.mode}] DONE  best_val_hard={best_val:.4f}  "
          f"wallclock={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
