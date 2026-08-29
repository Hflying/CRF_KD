"""CRF-KD package: knowledge distillation losses for CTC-CRF basecallers.

Rebuilt 2026-05-14 after source files were lost. Public API matches what
``scripts/train_r1_scale.py`` and ``scripts/kd_validate.py`` import.
"""

from .crf_kd import (
    CRFKDLoss,
    l_post_kd,
    l_post_joint_kd,
    l_fwd_kd,
    l_hard_kd,
)
from .filter import EntropyFilter, CurriculumScheduler, compute_chunk_entropy

__all__ = [
    "CRFKDLoss",
    "l_post_kd",
    "l_post_joint_kd",
    "l_fwd_kd",
    "l_hard_kd",
    "EntropyFilter",
    "CurriculumScheduler",
    "compute_chunk_entropy",
]
