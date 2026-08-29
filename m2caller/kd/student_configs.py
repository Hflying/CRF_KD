"""Student architecture presets + build_student helper for CRF-KD.

Each preset spec is fed to ``build_student(preset_name)`` to produce a
CRFStudent (Mamba2 backbone + Bonito linear-CRF head, state_len=5).
The stride-5 variants match Bonito r9.4.1 sup teacher's T axis so soft
KD losses (post / fwd) can be applied; stride-4 variants are kept for
M2Caller-compatible inference paths.

Rebuilt 2026-05-14 from project memory.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Presets (research_proposal.md §B.3.5)
# ---------------------------------------------------------------------------

M2_TINY: Dict[str, Any] = dict(
    name="m2_tiny",
    n_layers=2,
    d_model=128,
    state_len=5,
    stride=4,
    ssm_d_state=64,
    ssm_d_conv=4,
    ssm_headdim=64,
    alphabet="NACGT",
    scale=5.0,
)


M2_TINY_S5: Dict[str, Any] = dict(
    name="m2_tiny_s5",
    n_layers=2,
    d_model=128,
    state_len=5,
    stride=5,                 # Match Bonito r9.4.1 sup teacher T axis
    ssm_d_state=64,
    ssm_d_conv=4,
    ssm_headdim=64,
    alphabet="NACGT",
    scale=5.0,
)


M2_SMALL: Dict[str, Any] = dict(
    name="m2_small",
    n_layers=4,
    d_model=256,
    state_len=5,
    stride=4,
    ssm_d_state=128,
    ssm_d_conv=4,
    ssm_headdim=64,
    alphabet="NACGT",
    scale=5.0,
)


M2_SMALL_S5: Dict[str, Any] = dict(
    name="m2_small_s5",
    n_layers=4,
    d_model=256,
    state_len=5,
    stride=5,
    ssm_d_state=128,
    ssm_d_conv=4,
    ssm_headdim=64,
    alphabet="NACGT",
    scale=5.0,
)


PRESETS: Dict[str, Dict[str, Any]] = {
    "m2_tiny": M2_TINY,
    "m2_tiny_s5": M2_TINY_S5,
    "m2_small": M2_SMALL,
    "m2_small_s5": M2_SMALL_S5,
}


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class _SignalSampler(nn.Module):
    """Strided CNN signal down-sampler.

    stride=4: two stride-2 convs (M2Caller default).
    stride=5: Bonito-style three convs (k5-k5-k19 stride 1-1-5) so the
              output T = ceil(L/5) matches Bonito r9.4.1 sup teacher.
    """

    def __init__(self, in_ch: int = 1, d_model: int = 256, mid_ch: int = 64,
                 stride: int = 4):
        super().__init__()
        if stride == 4:
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_ch, d_model, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
            )
        elif stride == 5:
            # padding = ks // 2 → input 10000 yields output T=2000 (matches teacher).
            self.net = nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_ch, mid_ch, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_ch, d_model, kernel_size=19, stride=5, padding=9, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"stride must be 4 or 5, got {stride}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SimpleBackbone(nn.Module):
    """CPU-safe CNN-only fallback (used when mamba_ssm or CUDA unavailable)."""

    def __init__(self, d_model: int = 256, stride: int = 4):
        super().__init__()
        self.sampler = _SignalSampler(1, d_model, stride=stride)
        self.mixer = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        h = self.sampler(signal)
        h = self.mixer(h) + h
        h = h.transpose(1, 2)
        return self.norm(h)


class _Mamba2Backbone(nn.Module):
    """Mamba2-based backbone used for real training (requires CUDA)."""

    def __init__(self, d_model: int, n_layers: int,
                 ssm_d_state: int, ssm_d_conv: int, ssm_headdim: int,
                 stride: int = 4):
        super().__init__()
        try:
            from mamba_ssm.modules.mamba2 import Mamba2  # type: ignore
        except Exception:
            from mamba_ssm import Mamba2  # type: ignore
        self.sampler = _SignalSampler(1, d_model, stride=stride)
        self.blocks = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=ssm_d_state,
                   d_conv=ssm_d_conv, headdim=ssm_headdim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        h = self.sampler(signal)   # [B, C, L/stride]
        x = h.transpose(1, 2)      # [B, L', C]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Student wrapper
# ---------------------------------------------------------------------------

class CRFStudent(nn.Module):
    """Backbone + Bonito LinearCRFEncoder head.

    Exposes ``self.seqdist`` (a ``bonito.crf.model.CTC_CRF`` instance)
    so the KD losses can reuse the teacher's forward-backward machinery.

    forward(signal) -> (T, N, n_score) CRF scores.
    """

    def __init__(self, backbone: nn.Module, d_model: int, state_len: int,
                 alphabet: str = "NACGT", scale: float = 5.0):
        super().__init__()
        from bonito.nn import LinearCRFEncoder
        from bonito.crf.model import CTC_CRF

        self.backbone = backbone
        self.state_len = state_len
        self.alphabet = list(alphabet)
        self.n_base = len(self.alphabet) - 1
        self.head = LinearCRFEncoder(
            d_model, self.n_base, state_len,
            activation="tanh", scale=scale,
        )
        self.seqdist = CTC_CRF(state_len=state_len, alphabet=self.alphabet)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        # Accept both [B, 1, L] and [B, L, 1].
        if signal.dim() == 3 and signal.shape[1] != 1:
            signal = signal.transpose(1, 2)
        feat = self.backbone(signal)           # [B, L', d_model]
        feat = feat.transpose(0, 1).contiguous()  # [T, B, d_model]
        return self.head(feat)                 # [T, B, n_score]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_student(preset: str, backbone: str = "auto") -> CRFStudent:
    """Instantiate a student from a preset name.

    Args:
        preset: one of ``"m2_tiny"``, ``"m2_tiny_s5"``, ``"m2_small"``, ``"m2_small_s5"``.
        backbone: ``"auto"`` picks Mamba2 if importable AND CUDA available,
                  otherwise falls back to ``_SimpleBackbone``. Force with
                  ``"mamba2"`` or ``"simple"``.
    """
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Available: {list(PRESETS)}")
    cfg = PRESETS[preset]

    if backbone == "auto":
        use_mamba = torch.cuda.is_available()
        if use_mamba:
            try:
                import mamba_ssm  # noqa: F401
            except Exception:
                use_mamba = False
        backbone = "mamba2" if use_mamba else "simple"

    if backbone == "mamba2":
        bb = _Mamba2Backbone(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            ssm_d_state=cfg["ssm_d_state"],
            ssm_d_conv=cfg["ssm_d_conv"],
            ssm_headdim=cfg["ssm_headdim"],
            stride=cfg.get("stride", 4),
        )
    elif backbone == "simple":
        bb = _SimpleBackbone(d_model=cfg["d_model"], stride=cfg.get("stride", 4))
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")

    return CRFStudent(
        backbone=bb,
        d_model=cfg["d_model"],
        state_len=cfg["state_len"],
        alphabet=cfg["alphabet"],
        scale=cfg["scale"],
    )
