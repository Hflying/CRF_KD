"""
Minimal koi.ctc stub for Bonito CRF model loading and inference.

Implements CTC-CRF Viterbi decoding + JIT-accelerated forward/backward
in pure PyTorch (no proprietary CUDA kernels).

Rebuilt 2026-05-14 from project memory after source file was lost.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Semirings (lifted from Bonito's koi import surface)
# ---------------------------------------------------------------------------

class semiring:
    one = 0.0
    zero = float('-inf')


class Log(semiring):
    one = 0.0
    zero = float('-inf')

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def sum(a, dim=-1):
        return torch.logsumexp(a, dim)


class Max(semiring):
    one = 0.0
    zero = float('-inf')

    @staticmethod
    def mul(a, b):
        return a + b

    @staticmethod
    def sum(a, dim=-1):
        return a.max(dim).values


# ---------------------------------------------------------------------------
# Successor-table builders
# ---------------------------------------------------------------------------

def _build_inv_idx(idx):
    """Backward-compat (lossy) inverse-transition table.

    Returns ``(n_states, n_alpha)`` where ``inv_idx[s, x] = first successor of s via x``,
    or -1 if invalid. Only ONE of the ``n_base`` move successors per (s, x) is kept,
    which loses (n_base-1)/n_base of the backward flow — DO NOT use for new code.
    """
    n_states, n_alpha = idx.shape
    inv_idx = torch.full((n_states, n_alpha), -1, dtype=torch.long, device=idx.device)
    for x in range(n_alpha):
        for s_prime in range(n_states):
            s = idx[s_prime, x].item()
            if s >= 0:
                inv_idx[s, x] = s_prime
    return inv_idx


def _build_inv_succ_full(n_states: int, n_base: int, device=None):
    """Build the COMPLETE Bonito CTC-CRF inverse-transition table.

    For each state ``s`` of a state_len-k CRF over n_base bases:
      * 1 stay successor: s itself, via x=0
      * n_base move successors: every (suffix_of_s, b_new), each via
        x = b_1 + 1 where b_1 is the leading base of s.

    The lossy ``_build_inv_idx`` keeps only one of the n_base move successors
    per (s, x); this builder keeps all of them.

    Returns:
        succ_state: (n_states, 1 + n_base) int64 — successor state index
        succ_x:     (n_states, 1 + n_base) int64 — transition x at successor
    """
    state_len = int(round(math.log(n_states) / math.log(n_base)))
    assert n_base ** state_len == n_states, \
        f"n_states ({n_states}) is not n_base^state_len for n_base={n_base}"
    pow_km1 = n_base ** (state_len - 1)

    s = torch.arange(n_states, dtype=torch.long, device=device)
    b1 = s // pow_km1                          # leading base (0..n_base-1)
    s_succ_base = (s % pow_km1) * n_base       # offset for move successors

    succ_state = torch.empty(n_states, 1 + n_base, dtype=torch.long, device=device)
    succ_x = torch.empty(n_states, 1 + n_base, dtype=torch.long, device=device)

    # Stay slot
    succ_state[:, 0] = s
    succ_x[:, 0] = 0

    # Move slots
    for b_new in range(n_base):
        succ_state[:, 1 + b_new] = s_succ_base + b_new
        succ_x[:, 1 + b_new] = b1 + 1          # all 4 moves from s use the same x=b1+1

    return succ_state, succ_x


# ---------------------------------------------------------------------------
# Viterbi decoding
# ---------------------------------------------------------------------------

@torch.jit.script
def _viterbi_step(alpha: torch.Tensor, Ms_t: torch.Tensor,
                  idx_exp: torch.Tensor, idx_dev: torch.Tensor):
    """Single Viterbi forward step (JIT-compiled).

    Ms_t[n, s_prime, x] = score of transition emitting x ending at state s_prime.
    alpha[t+1, s_prime] = max_x ( alpha[t, idx[s_prime, x]] + Ms_t[n, s_prime, x] ).
    """
    alpha_prev = alpha[:, idx_dev]                   # (N, n_states, n_alpha)
    edge = alpha_prev + Ms_t                         # (N, n_states, n_alpha)
    return edge.max(dim=-1).values, edge.argmax(dim=-1)


def _viterbi_forward(Ms, idx_dev, N, T, n_states, n_alpha):
    """Viterbi forward pass; returns (alpha_T, back)."""
    alpha = torch.zeros(N, n_states, device=Ms.device)
    back = torch.zeros(T, N, n_states, dtype=torch.long, device=Ms.device)
    idx_exp = idx_dev.unsqueeze(0).expand(N, -1, -1)
    for t in range(T):
        alpha, best = _viterbi_step(alpha, Ms[t], idx_exp, idx_dev)
        back[t] = best
    return alpha, back


def viterbi_decode_from_scores(scores, idx):
    """Legacy Viterbi decode. Returns list of N int lists where each value is
    the transition type x (0=stay, 1..n_alpha-1=move variant) at each time t.

    NOTE: x is the transition INDEX, not the emitted base — use
    ``viterbi_decode_with_states`` for emission-level decode.
    """
    T, N, n_score = scores.shape
    idx_dev = idx.long().to(scores.device)
    n_states, n_alpha = idx_dev.shape

    Ms = scores.reshape(T, N, n_states, n_alpha).float()
    alpha_T, back = _viterbi_forward(Ms, idx_dev, N, T, n_states, n_alpha)

    back_cpu = back.cpu()
    idx_cpu = idx_dev.cpu()
    alpha_T_cpu = alpha_T.cpu()

    sequences = []
    for n in range(N):
        state = int(alpha_T_cpu[n].argmax().item())
        path_x = []
        for t in range(T - 1, -1, -1):
            x = int(back_cpu[t, n, state].item())
            path_x.append(x)
            state = int(idx_cpu[state, x].item())
        path_x.reverse()
        sequences.append(path_x)

    return sequences


def viterbi_decode_with_states(scores, idx, n_base):
    """Full Viterbi decode returning emitted bases directly.

    Each Bonito CRF move (x != 0) emits the base corresponding to
    ``1 + (dest_state % n_base)`` (1-indexed into the alphabet ['N','A','C','G','T']).

    Returns: list of N int lists. Each value is:
        0           — stay (no base emitted)
        1..n_base   — base index into the alphabet (matches alphabet[1+k])
    """
    T, N, n_score = scores.shape
    idx_dev = idx.long().to(scores.device)
    n_states, n_alpha = idx_dev.shape

    Ms = scores.reshape(T, N, n_states, n_alpha).float()
    alpha_T, back = _viterbi_forward(Ms, idx_dev, N, T, n_states, n_alpha)

    back_cpu = back.cpu()
    idx_cpu = idx_dev.cpu()
    alpha_T_cpu = alpha_T.cpu()

    sequences = []
    for n in range(N):
        state = int(alpha_T_cpu[n].argmax().item())
        emissions = []
        for t in range(T - 1, -1, -1):
            x = int(back_cpu[t, n, state].item())
            if x == 0:
                emissions.append(0)
            else:
                emissions.append(1 + (state % n_base))
            state = int(idx_cpu[state, x].item())
        emissions.reverse()
        sequences.append(emissions)

    return sequences


# ---------------------------------------------------------------------------
# SequenceDist stub (Bonito imports this name)
# ---------------------------------------------------------------------------

class SequenceDist(nn.Module):
    """Abstract base class stub matching what Bonito's CRF model imports.

    Bonito's ``CTC_CRF.viterbi`` calls ``self.posteriors(scores, Max)`` and
    expects something it can ``.argmax(2)``. Since the real koi computes
    proper forward-backward posteriors and we don't have that here, this
    stub returns scores unchanged — which makes the resulting "Viterbi"
    a per-timestep greedy argmax. ``CTC_CRF.ctc_loss`` etc. are NOT used
    via this path; CRF-KD uses the explicit forward/backward functions
    below instead.
    """

    def posteriors(self, scores, S=None):
        return scores


def logZ_cu(stay_scores, move_scores, target_lengths):
    raise NotImplementedError("logZ_cu is not available (koi stub)")


def viterbi_alignments(stay_scores, move_scores, target_lengths):
    raise NotImplementedError("viterbi_alignments is not available (koi stub)")


# ---------------------------------------------------------------------------
# JIT-accelerated sparse forward / backward / logZ
# ---------------------------------------------------------------------------

@torch.jit.script
def _logZ_loop(Ms: torch.Tensor, idx_dev: torch.Tensor, alpha_0: torch.Tensor,
               beta_T: torch.Tensor):
    """JIT-compiled forward log-Z. Ms: (T,N,S,A), idx: (S,A), alpha_0/beta_T: (N,S)."""
    T = Ms.size(0)
    alpha = alpha_0
    for t in range(T):
        alpha_prev = alpha.index_select(1, idx_dev.reshape(-1)).reshape(
            alpha.size(0), idx_dev.size(0), idx_dev.size(1))
        edge = alpha_prev + Ms[t]
        alpha = torch.logsumexp(edge, dim=-1)
    return torch.logsumexp(alpha + beta_T, dim=-1)


def logZ_cu_sparse(Ms, idx, alpha_0, beta_T, S=None):
    """JIT-accelerated sparse forward-backward log-Z."""
    idx_dev = idx.long().to(Ms.device)
    return _logZ_loop(Ms.float().contiguous(), idx_dev,
                      alpha_0.float().contiguous(), beta_T.float().contiguous())


@torch.jit.script
def _bwd_loop(Ms: torch.Tensor, succ_state: torch.Tensor, succ_x: torch.Tensor,
              beta_T: torch.Tensor):
    """JIT-compiled backward scores using the COMPLETE successor table.

    Ms: (T, N, S, A) — CRF scores
    succ_state, succ_x: (S, K) where K = 1 + n_base; the K successors per state
    beta_T: (N, S) — terminal backward scores

    Recursion:
        β_t(s) = log Σ_k exp( M_t(succ_state[s,k], succ_x[s,k]) + β_{t+1}(succ_state[s,k]) )

    Returns betas: (T+1, N, S).
    """
    T = Ms.size(0)
    N = Ms.size(1)
    S = Ms.size(2)
    A = Ms.size(3)
    K = succ_state.size(1)

    betas = torch.empty(T + 1, N, S, dtype=torch.float32, device=Ms.device)
    betas[T] = beta_T

    succ_lin = (succ_state * A + succ_x).reshape(-1)         # (S * K,)
    succ_state_flat = succ_state.reshape(-1)                  # (S * K,)

    for t in range(T - 1, -1, -1):
        Ms_flat = Ms[t].reshape(N, S * A)                     # (N, S*A)
        m = Ms_flat.index_select(1, succ_lin).reshape(N, S, K)
        b = betas[t + 1].index_select(1, succ_state_flat).reshape(N, S, K)
        edge = m + b
        betas[t] = torch.logsumexp(edge, dim=-1)

    return betas


def bwd_scores_cu_sparse(Ms, idx, beta_T, S=None, K=1):
    """JIT-accelerated sparse backward scores using the full successor table.

    Note: the ``idx`` argument is kept only for API compatibility — we now derive
    successor structure directly from Bonito's CRF state encoding.
    """
    T, N, n_states, n_alpha = Ms.shape
    n_base = n_alpha - 1
    succ_state, succ_x = _build_inv_succ_full(n_states, n_base, device=Ms.device)
    return _bwd_loop(Ms.float().contiguous(), succ_state, succ_x,
                     beta_T.float().contiguous())


@torch.jit.script
def _fwd_loop(Ms: torch.Tensor, idx_dev: torch.Tensor, alpha_0: torch.Tensor):
    """JIT-compiled forward scores. Returns alphas (T+1, N, S)."""
    T = Ms.size(0)
    N = Ms.size(1)
    S = Ms.size(2)
    alphas = torch.empty(T + 1, N, S, dtype=torch.float32, device=Ms.device)
    alphas[0] = alpha_0

    for t in range(T):
        alpha_prev = alphas[t].index_select(1, idx_dev.reshape(-1)).reshape(
            N, idx_dev.size(0), idx_dev.size(1))
        edge = alpha_prev + Ms[t]
        alphas[t + 1] = torch.logsumexp(edge, dim=-1)

    return alphas


def fwd_scores_cu_sparse(Ms, idx, alpha_0, S=None, K=1):
    """JIT-accelerated sparse forward scores."""
    idx_dev = idx.long().to(Ms.device)
    return _fwd_loop(Ms.float().contiguous(), idx_dev, alpha_0.float().contiguous())
