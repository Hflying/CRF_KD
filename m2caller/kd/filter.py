"""
Confidence filtering & curriculum for CRF-KD (Direction B §B.3.3-B.3.4).

* ``compute_chunk_entropy(teacher_scores, seqdist)``
    Mean entropy of the teacher's transition posterior over time for each
    chunk in the batch. Returns a (N,) tensor.

* ``EntropyFilter``
    Maintains a running estimate of the entropy distribution and produces
    a boolean keep-mask for each batch. Chunks whose mean-entropy exceeds
    the current percentile threshold are dropped.

* ``CurriculumScheduler``
    Linearly opens the acceptance percentile during training, starting at
    ``start_percentile`` (default 50) and ramping to ``end_percentile``
    (default 100) over ``warmup_epochs``.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch


EPS = 1e-12


def compute_chunk_entropy(teacher_scores: torch.Tensor, seqdist) -> torch.Tensor:
    """Per-chunk mean entropy of CRF transition posteriors.

    Args:
        teacher_scores: (T, N, n_score) raw teacher scores.
        seqdist: ``bonito.crf.model.CTC_CRF`` matching teacher.

    Returns:
        (N,) float tensor, mean over time of
        ``-sum_{s,x} P_t(s,x) log P_t(s,x)``.
    """
    with torch.no_grad():
        betas = seqdist.backward_scores(teacher_scores)
        trans, _ = seqdist.compute_transition_probs(teacher_scores, betas)  # (T, N, S, A)
        p = trans.clamp_min(EPS)
        h = -(p * p.log()).sum(dim=(-2, -1))   # (T, N)
        return h.mean(dim=0)                   # (N,)


class EntropyFilter:
    """Rolling-percentile filter for per-chunk teacher entropies.

    The filter stores up to ``window`` recent entropies and estimates the
    keep-threshold as the given percentile of that buffer. A chunk passes
    (True in the returned mask) if its entropy is at or below the threshold.

    If fewer than ``warmup_samples`` values have been seen, the filter is a
    no-op (returns all-True), so early batches are never dropped.
    """

    def __init__(self,
                 percentile_threshold: float = 80.0,
                 window: int = 4096,
                 warmup_samples: int = 32):
        self.percentile_threshold = float(percentile_threshold)
        self.window = int(window)
        self.warmup_samples = int(warmup_samples)
        self._buffer: "deque[float]" = deque(maxlen=self.window)
        self.threshold: Optional[float] = None

    def set_percentile(self, pct: float) -> None:
        self.percentile_threshold = float(pct)
        self._recompute_threshold()

    def _recompute_threshold(self) -> None:
        if len(self._buffer) < self.warmup_samples:
            self.threshold = None
            return
        q = self.percentile_threshold / 100.0
        vals = torch.tensor(list(self._buffer), dtype=torch.float32)
        self.threshold = float(torch.quantile(vals, q).item())

    def update_threshold(self, entropies_from_batch: torch.Tensor) -> None:
        """Ingest a batch of entropies and refresh the keep-threshold."""
        for v in entropies_from_batch.detach().cpu().flatten().tolist():
            self._buffer.append(float(v))
        self._recompute_threshold()

    def __call__(self, entropies: torch.Tensor) -> torch.Tensor:
        """Return a bool mask (True = keep)."""
        if self.threshold is None:
            return torch.ones_like(entropies, dtype=torch.bool)
        return entropies <= self.threshold


class CurriculumScheduler:
    """Linear curriculum over the acceptance percentile.

    ``start_percentile`` at epoch 0 -> ``end_percentile`` at
    ``warmup_epochs``; afterwards the percentile stays at ``end_percentile``
    for the remainder of training.
    """

    def __init__(self,
                 total_epochs: int,
                 warmup_epochs: int = 30,
                 start_percentile: int = 50,
                 end_percentile: int = 100):
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = max(int(warmup_epochs), 1)
        self.start_percentile = int(start_percentile)
        self.end_percentile = int(end_percentile)

    def current_percentile(self, epoch: int) -> int:
        if epoch >= self.warmup_epochs:
            return self.end_percentile
        frac = max(0, epoch) / self.warmup_epochs
        val = self.start_percentile + (self.end_percentile - self.start_percentile) * frac
        return int(round(val))
