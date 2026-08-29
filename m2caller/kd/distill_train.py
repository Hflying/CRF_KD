"""
CRF-KD distillation training entry point (Direction B, research_proposal.md).

Pipeline
--------
1. Load Bonito teacher (frozen, ``.eval()``). Teacher forward runs inside
   ``torch.no_grad()`` in the training loop.
2. Build a student from a preset (``m2_small`` / ``m2_tiny``).
3. Stream unlabeled signal chunks from ``m2caller.utils.dataset`` - ground
   truth labels are not required since the teacher supplies them via Viterbi.
4. For each batch:
       with torch.no_grad():
           teacher_scores  = teacher(signal).float()
           teacher_targets = viterbi_decode_with_states(teacher_scores, ...)
       student_scores = student(signal).float()
       loss_dict = kd_loss(teacher_scores, student_scores, teacher_targets, ...)
       loss_dict['total'].backward()

Smoke test
----------
``python -m m2caller.kd.distill_train --smoke_test`` runs a single optimizer
step on a dummy signal batch using a CPU-only student + fabricated teacher
scores. It exercises all three losses (with matching state_len) and the
state_len-mismatch fallback, then prints the final loss.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m2caller.kd.crf_kd import CRFKDLoss
from m2caller.kd.filter import (
    EntropyFilter,
    CurriculumScheduler,
    compute_chunk_entropy,
)
from m2caller.kd.student_configs import PRESETS, build_student


# ---------------------------------------------------------------------------
# Teacher decoding
# ---------------------------------------------------------------------------

def teacher_viterbi_targets(teacher_scores: torch.Tensor,
                            seqdist) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the teacher's Viterbi to obtain hard labels (N, L), lengths (N,).

    Uses our pure-PyTorch ``koi.ctc.viterbi_decode_with_states`` (since the
    stub's ``posteriors`` does not compute true Viterbi).
    """
    from koi.ctc import viterbi_decode_with_states

    emissions = viterbi_decode_with_states(
        teacher_scores, seqdist.idx, seqdist.n_base,
    )
    seqs: List[List[int]] = [[b for b in e if b != 0] for e in emissions]
    lengths = [len(s) for s in seqs]

    # Need L >= state_len so prepare_ctc_scores yields at least one lattice
    # position (n = L - state_len + 1).
    min_len = seqdist.state_len + 1
    L = max(min_len, max(lengths) if lengths else min_len)
    N = len(seqs)
    targets = torch.zeros(N, L, dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            targets[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return targets, torch.tensor(lengths, dtype=torch.int32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test() -> float:
    """Build a tiny CPU student, fake teacher scores, backprop one step."""
    print("[smoke] building tiny student (CPU, simple backbone)", flush=True)
    torch.manual_seed(0)
    device = torch.device("cpu")

    student = build_student("m2_tiny", backbone="simple").to(device)
    student.train()

    # Dummy signal batch: [B=2, 1, L=512]  -> L/4 = 128 timesteps.
    B, L = 2, 512
    signal = torch.randn(B, 1, L, device=device)

    student_scores = student(signal).float()
    T, N, C = student_scores.shape
    print(f"[smoke] student output: T={T} N={N} n_score={C} "
          f"(state_len={student.state_len}, n_base={student.n_base})",
          flush=True)

    # Fabricate teacher scores matching student's shape (same state_len).
    torch.manual_seed(1)
    teacher_scores = torch.randn(T, N, C, device=device) * 2.0

    # Teacher Viterbi decode -> pseudo targets.
    teacher_targets, target_lengths = teacher_viterbi_targets(
        teacher_scores.detach(), student.seqdist,
    )
    print(f"[smoke] teacher targets shape={tuple(teacher_targets.shape)} "
          f"lengths={target_lengths.tolist()}", flush=True)

    kd_loss = CRFKDLoss(lambda_post=1.0, lambda_fwd=0.5, lambda_hard=1.0)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)

    losses = kd_loss(
        teacher_scores=teacher_scores,
        student_scores=student_scores,
        teacher_targets=teacher_targets,
        target_lengths=target_lengths,
        teacher_seqdist=student.seqdist,   # same state_len
        student_seqdist=student.seqdist,
    )
    print(f"[smoke] losses: "
          f"total={losses['total'].item():.4f} "
          f"post={losses['post'].item():.4f} "
          f"fwd={losses['fwd'].item():.4f} "
          f"hard={losses['hard'].item():.4f}", flush=True)

    opt.zero_grad()
    losses["total"].backward()
    opt.step()

    # Verify state_len-mismatch fallback (warning + only L_hard).
    class _FakeSeqdist:
        state_len = 3
        alphabet = ["N", "A", "C", "G", "T"]
    kd_loss_mm = CRFKDLoss(lambda_post=1.0, lambda_fwd=0.5, lambda_hard=1.0)
    losses_mm = kd_loss_mm(
        teacher_scores=teacher_scores,
        student_scores=student_scores,
        teacher_targets=teacher_targets,
        target_lengths=target_lengths,
        teacher_seqdist=_FakeSeqdist(),
        student_seqdist=student.seqdist,
    )
    print(f"[smoke] mismatched state_len fallback: "
          f"post={losses_mm['post'].item():.4f} "
          f"fwd={losses_mm['fwd'].item():.4f} "
          f"hard={losses_mm['hard'].item():.4f}", flush=True)

    # Exercise the entropy filter + curriculum scheduler.
    entropies = compute_chunk_entropy(teacher_scores, student.seqdist)
    ef = EntropyFilter(percentile_threshold=80, warmup_samples=1)
    ef.update_threshold(entropies)
    keep = ef(entropies)
    cs = CurriculumScheduler(total_epochs=60, warmup_epochs=30)
    print(f"[smoke] entropies={entropies.tolist()} "
          f"threshold={ef.threshold} keep={keep.tolist()}", flush=True)
    print(f"[smoke] curriculum: epoch 0 -> {cs.current_percentile(0)}%, "
          f"epoch 15 -> {cs.current_percentile(15)}%, "
          f"epoch 40 -> {cs.current_percentile(40)}%", flush=True)

    print(f"[smoke] final loss: {losses['total'].item():.6f}", flush=True)
    print("[smoke] OK", flush=True)
    return float(losses["total"].item())


# ---------------------------------------------------------------------------
# Full training (scaffolding only; not exercised by smoke test)
# ---------------------------------------------------------------------------

def run_full_training(args: argparse.Namespace) -> None:  # pragma: no cover
    """Skeleton for real distillation training.

    Intentionally minimal - the project convention is that real training
    scripts live under ``scripts/`` and are launched with explicit configs.
    """
    from bonito.util import load_model
    from m2caller.utils.dataset import NumpyRecordsDataset, collate_variable_windows

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[kd] device={device}")

    print(f"[kd] loading teacher from {args.teacher_model}")
    teacher = load_model(args.teacher_model, device=str(device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[kd] building student preset={args.student_preset}")
    student = build_student(args.student_preset).to(device)

    if teacher.seqdist.state_len != student.seqdist.state_len:
        warnings.warn(
            f"[kd] Teacher state_len={teacher.seqdist.state_len} vs "
            f"student state_len={student.seqdist.state_len}; "
            f"L_post and L_fwd will be disabled in CRFKDLoss.",
            RuntimeWarning,
        )

    ds = NumpyRecordsDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=True,
                        collate_fn=collate_variable_windows)

    kd_loss = CRFKDLoss(
        lambda_post=args.lambda_post,
        lambda_fwd=args.lambda_fwd,
        lambda_hard=args.lambda_hard,
    ).to(device)

    ef = EntropyFilter(percentile_threshold=80) if args.use_entropy_filter else None
    cs = (CurriculumScheduler(total_epochs=args.epochs, warmup_epochs=min(30, args.epochs))
          if args.use_curriculum else None)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if cs is not None and ef is not None:
            ef.set_percentile(cs.current_percentile(epoch))

        for _read_ids, signal, _lengths in loader:
            signal = signal.to(device)
            # Expect signal [N, W, 1] -> [N, 1, W]
            sig = signal.squeeze(-1).unsqueeze(1) if signal.dim() == 3 else signal

            with torch.no_grad():
                teacher_scores = teacher(sig).float()
                teacher_targets, target_lengths = teacher_viterbi_targets(
                    teacher_scores, teacher.seqdist,
                )

            if ef is not None:
                entropies = compute_chunk_entropy(teacher_scores, teacher.seqdist)
                ef.update_threshold(entropies)
                keep = ef(entropies)
                if keep.any():
                    sig = sig[keep]
                    teacher_scores = teacher_scores[:, keep]
                    teacher_targets = teacher_targets[keep.cpu()]
                    target_lengths = target_lengths[keep.cpu()]
                else:
                    continue

            student_scores = student(sig).float()
            losses = kd_loss(
                teacher_scores, student_scores, teacher_targets, target_lengths,
                teacher_seqdist=teacher.seqdist,
                student_seqdist=student.seqdist,
            )
            opt.zero_grad()
            losses["total"].backward()
            opt.step()

            print(f"[kd] epoch={epoch} total={losses['total'].item():.4f} "
                  f"post={losses['post'].item():.4f} "
                  f"fwd={losses['fwd'].item():.4f} "
                  f"hard={losses['hard'].item():.4f}",
                  flush=True)

        ckpt = os.path.join(args.out_dir, f"student_epoch{epoch:03d}.pt")
        torch.save({"student": student.state_dict(), "epoch": epoch}, ckpt)
        print(f"[kd] saved {ckpt}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="CRF-KD distillation (Bonito sup -> M2Caller student).",
    )
    ap.add_argument("--teacher_model",
                    default="tools/bonito_models/dna_r9.4.1_e8_sup@v3.3")
    ap.add_argument("--student_preset", default="m2_small",
                    choices=list(PRESETS.keys()))
    ap.add_argument("--data_dir", default="data/wick/train_extracted")
    ap.add_argument("--out_dir", default="checkpoints_kd/m2_small")
    ap.add_argument("--lambda_post", type=float, default=1.0)
    ap.add_argument("--lambda_fwd", type=float, default=0.5)
    ap.add_argument("--lambda_hard", type=float, default=1.0)
    ap.add_argument("--use_entropy_filter", action="store_true")
    ap.add_argument("--use_curriculum", action="store_true")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--smoke_test", action="store_true",
                    help="Run a CPU-only one-step sanity check and exit.")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.smoke_test:
        run_smoke_test()
        return 0
    run_full_training(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
