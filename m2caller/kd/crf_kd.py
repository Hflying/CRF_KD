"""
CRF-KD losses (Direction B / ICASSP'27 paper).

Implements four loss components used in the CRF-KD recipe:

* L_post (β-only conditional) — KL between teacher / student transition
  posteriors P_t(x | s_dest) obtained from Bonito's compute_transition_probs.
  This is the natural-first-attempt KD target. EMPIRICALLY HARMFUL at
  validation scale (-17pp); kept for ablation.

* L_post_joint — KL between TRUE joint transition posteriors
  P*_t(s, x | obs) derived from full forward-backward (both α and β).
  This is the winning target (+27pp at validation, +0.7pp at 50-iso scale).

* L_fwd — MSE between per-timestep normalised forward log-scores α_t(s).
  Empirically destabilises optimisation; not used in final recipe.

* L_hard — Standard CTC-CRF NLL using the teacher's Viterbi decode as
  pseudo-labels. Uses a pure-PyTorch logZ since the koi stub's logZ_cu
  is NotImplementedError.

Rebuilt 2026-05-14 from project memory.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn


EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _teacher_trans_probs(scores: torch.Tensor, seqdist) -> torch.Tensor:
    """β-only transition posteriors P_t(s, x), shape (T, N, n_states, n_alpha).

    Uses Bonito's existing forward-backward via the supplied seqdist:
      betas = seqdist.backward_scores(scores)
      trans, _ = seqdist.compute_transition_probs(scores, betas)
    """
    betas = seqdist.backward_scores(scores)
    trans, _ = seqdist.compute_transition_probs(scores, betas)
    return trans


def _normalise_alpha(alphas: torch.Tensor) -> torch.Tensor:
    """Row-normalise forward log-scores over states at every time step."""
    return alphas - torch.logsumexp(alphas, dim=-1, keepdim=True)


def _logZ_target_forward(stay: torch.Tensor,
                         move: torch.Tensor,
                         seq_lengths: torch.Tensor,
                         neg_large: float = -1e4) -> torch.Tensor:
    """Pure-PyTorch CTC-CRF target-constrained forward.

    Mirrors koi's logZ_cu semantics: at every time step a path may either
    stay at position p or move p-1 -> p.

    Implementation note: we use ``-1e4`` for unreachable initial positions
    instead of ``-inf``. ``torch.logsumexp`` with two ``-inf`` inputs
    propagates NaN gradients (0/0 local sensitivity); ``-1e4`` is small
    enough that ``exp(-1e4)`` underflows to 0 forward but keeps the
    backward graph finite.

    Args:
        stay: (T, N, Lp)    log-scores for staying at position p
        move: (T, N, Lp-1)  log-scores for moving p-1 -> p
        seq_lengths: (N,)   number of k-mer positions per chunk
                            (= target_lengths + 1 - state_len)
    Returns:
        (N,) logZ for each chunk.
    """
    T, N, Lp = stay.shape
    alpha = stay.new_full((N, Lp), neg_large)
    alpha[:, 0] = 0.0

    for t in range(T):
        stay_a = alpha + stay[t]                                # (N, Lp)
        move_a = stay.new_full((N, Lp), neg_large)
        if Lp > 1:
            move_a[:, 1:] = alpha[:, :-1] + move[t]
        alpha = torch.logsumexp(torch.stack([stay_a, move_a], dim=0), dim=0)

    idx = (seq_lengths.long() - 1).clamp_min(0).to(alpha.device).unsqueeze(1)
    return alpha.gather(1, idx).squeeze(1)


def _joint_log_posteriors(scores: torch.Tensor, seqdist):
    """Compute true joint transition log-posteriors log P(s, x | t).

    Uses BOTH forward α and backward β scores:
      log P(t, s, x) = α_t(idx[s,x]) + M_t(s, x) + β_{t+1}(s) − log Z

    Returns log_p of shape (T, N, n_states, n_alpha); per (t, n) slice,
    exp(log_p) sums to 1 over (s, x).
    """
    T, N, _ = scores.shape
    idx = seqdist.idx.long().to(scores.device)
    n_states, n_alpha = idx.shape
    Ms = scores.reshape(T, N, n_states, n_alpha)

    alphas = seqdist.forward_scores(scores)            # (T+1, N, n_states)
    betas = seqdist.backward_scores(scores)            # (T+1, N, n_states)

    # alpha at predecessor state via idx[s, x] — fancy indexing
    alpha_pred = alphas[:-1].index_select(-1, idx.reshape(-1)).reshape(
        T, N, n_states, n_alpha
    )
    beta_dest = betas[1:].unsqueeze(-1)                # (T, N, n_states, 1)
    log_unnorm = alpha_pred + Ms + beta_dest           # (T, N, n_states, n_alpha)

    # Normalise over (s, x) per (t, n).
    flat = log_unnorm.flatten(start_dim=2)             # (T, N, n_states*n_alpha)
    Z = torch.logsumexp(flat, dim=-1)                  # (T, N)
    return log_unnorm - Z[:, :, None, None]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def l_post_kd(teacher_scores: torch.Tensor,
              student_scores: torch.Tensor,
              seqdist) -> torch.Tensor:
    """β-only conditional transition-posterior KL (L_post variant 1).

    The trans_probs returned by ``compute_transition_probs`` have shape
    (T, N, n_states, n_alpha) where each (t, n, s) row sums to 1 over
    n_alpha (conditional on dest state s). Summing over both s and x in
    the KL would multiply per-row KL by n_states (=1024 for state_len=5).
    We average over n_states instead so the loss is on the same scale as
    L_hard.

    EMPIRICALLY HARMFUL: matching this conditional pulls a small student
    into a parameter regime where Viterbi argmax produces near-random
    output. Kept for ablation; prefer ``l_post_joint_kd``.
    """
    with torch.no_grad():
        p_teacher = _teacher_trans_probs(teacher_scores.detach(), seqdist)

    betas_s = seqdist.backward_scores(student_scores)
    p_student, _ = seqdist.compute_transition_probs(student_scores, betas_s)

    log_t = (p_teacher + EPS).log()
    log_s = (p_student + EPS).log()
    kl_per_row = (p_teacher * (log_t - log_s)).sum(dim=-1)   # (T, N, n_states)
    return kl_per_row.mean()


def l_post_joint_kd(teacher_scores: torch.Tensor,
                    student_scores: torch.Tensor,
                    seqdist) -> torch.Tensor:
    """**Joint** posterior KL — the WINNING formulation (CRF-KD).

    Matches the full joint P*_t(s, x | obs) derived from BOTH α and β,
    which is what Viterbi argmax actually optimises. Empirically +27 pp
    over hard-target-only at validation scale (vs -17 pp for β-only).
    """
    with torch.no_grad():
        log_p_t = _joint_log_posteriors(teacher_scores.detach(), seqdist)

    log_p_s = _joint_log_posteriors(student_scores, seqdist)

    p_t = log_p_t.exp()
    kl_per_step = (p_t * (log_p_t - log_p_s)).sum(dim=(-2, -1))   # (T, N)
    return kl_per_step.mean()


def l_fwd_kd(teacher_scores: torch.Tensor,
             student_scores: torch.Tensor,
             seqdist) -> torch.Tensor:
    """Forward-score MSE (DEPRECATED — destabilises optimisation).

    Kept only for ablation; not used in CRF-KD final formulation.
    """
    with torch.no_grad():
        a_teacher = _normalise_alpha(seqdist.forward_scores(teacher_scores.detach()))

    a_student = _normalise_alpha(seqdist.forward_scores(student_scores))
    return torch.mean((a_teacher - a_student) ** 2)


def l_hard_kd(student_scores: torch.Tensor,
              teacher_targets: torch.Tensor,
              target_lengths: torch.Tensor,
              seqdist) -> torch.Tensor:
    """CTC-CRF NLL on the teacher's Viterbi-decoded targets (L_hard).

    Uses seqdist.normalise + seqdist.prepare_ctc_scores + a pure-PyTorch
    target-constrained forward (since koi stub's logZ_cu is unavailable).

    Args:
        student_scores:  (T, N, n_score)
        teacher_targets: (N, L) long — 0 pad, 1..n_base base indices.
        target_lengths:  (N,) int — true length of each target (<= L).
        seqdist:         student's CTC_CRF instance.
    """
    scores = seqdist.normalise(student_scores)
    stay, move = seqdist.prepare_ctc_scores(scores, teacher_targets.to(scores.device))
    seq_lens = target_lengths.to(scores.device).long() + 1 - seqdist.state_len
    seq_lens = seq_lens.clamp_min(1)
    logz = _logZ_target_forward(stay, move, seq_lens)
    loss = -(logz / target_lengths.to(logz.device).clamp_min(1).float())
    return loss.mean()


# ---------------------------------------------------------------------------
# Combined module
# ---------------------------------------------------------------------------

class CRFKDLoss(nn.Module):
    """Combined CRF-KD loss.

    Default config matches the ICASSP'27 final recipe:
      lambda_post=1.0  (joint posterior),
      lambda_fwd=0.0   (dropped — was destabilising),
      lambda_hard=1.0.

    forward() returns a dict with keys 'total', 'post', 'fwd', 'hard'.

    Args:
      joint_post: if True use ``l_post_joint_kd`` (P3 winner),
                  if False use ``l_post_kd`` (β-only, ablation/legacy).
      require_same_state_len: if True, raise on teacher/student state_len
                  mismatch; else silently fall back to L_hard-only with a
                  one-time RuntimeWarning.
    """

    def __init__(self,
                 lambda_post: float = 1.0,
                 lambda_fwd: float = 0.0,
                 lambda_hard: float = 1.0,
                 require_same_state_len: bool = False,
                 joint_post: bool = False):
        super().__init__()
        self.lambda_post = float(lambda_post)
        self.lambda_fwd = float(lambda_fwd)
        self.lambda_hard = float(lambda_hard)
        self.require_same_state_len = require_same_state_len
        self.joint_post = bool(joint_post)
        self._warned_mismatch = False

    def _soft_losses_enabled(self, teacher_seqdist, student_seqdist) -> bool:
        same = (teacher_seqdist.state_len == student_seqdist.state_len and
                len(teacher_seqdist.alphabet) == len(student_seqdist.alphabet))
        if same:
            return True
        if self.require_same_state_len:
            raise ValueError(
                f"Teacher/student state_len mismatch "
                f"(T={teacher_seqdist.state_len} vs S={student_seqdist.state_len}); "
                f"require_same_state_len=True."
            )
        if not self._warned_mismatch:
            warnings.warn(
                f"[CRFKDLoss] Teacher state_len={teacher_seqdist.state_len} != "
                f"Student state_len={student_seqdist.state_len}; "
                f"L_post and L_fwd will be skipped and only L_hard is used.",
                RuntimeWarning,
            )
            self._warned_mismatch = True
        return False

    def forward(self,
                teacher_scores: torch.Tensor,
                student_scores: torch.Tensor,
                teacher_targets: torch.Tensor,
                target_lengths: torch.Tensor,
                teacher_seqdist,
                student_seqdist) -> Dict[str, torch.Tensor]:
        device = student_scores.device
        zero = torch.zeros((), device=device)

        soft_ok = self._soft_losses_enabled(teacher_seqdist, student_seqdist)

        if soft_ok and self.lambda_post > 0:
            if self.joint_post:
                post = l_post_joint_kd(teacher_scores, student_scores, student_seqdist)
            else:
                post = l_post_kd(teacher_scores, student_scores, student_seqdist)
        else:
            post = zero

        if soft_ok and self.lambda_fwd > 0:
            fwd = l_fwd_kd(teacher_scores, student_scores, student_seqdist)
        else:
            fwd = zero

        if self.lambda_hard > 0:
            hard = l_hard_kd(student_scores, teacher_targets, target_lengths,
                             student_seqdist)
        else:
            hard = zero

        total = (self.lambda_post * post +
                 self.lambda_fwd * fwd +
                 self.lambda_hard * hard)

        return {"total": total, "post": post, "fwd": fwd, "hard": hard}
