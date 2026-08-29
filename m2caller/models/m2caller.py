import math, torch, torch.nn as nn, torch.nn.functional as F
from m2caller.utils.constants import VOCAB_SIZE, BLANK

class SignalSampler(nn.Module):
    def __init__(self, in_ch=1, d_model=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, d_model//2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model//2, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):  # x: [B, 1, L]
        return self.net(x)  # [B, C, L']

class Mamba2LikeBlock(nn.Module):
    """A lightweight selective state-space block.
    Implements h_t = A*h_{t-1} + B*x_t with learned gates and depthwise conv mixing.
    Complexity: O(L).
    """
    def __init__(self, d_model, ssm_dim):
        super().__init__()
        self.d_model = d_model
        self.ssm_dim = ssm_dim
        self.in_proj = nn.Linear(d_model, ssm_dim)
        self.A = nn.Parameter(torch.randn(ssm_dim) * -0.01)  # stable (negative)
        self.B = nn.Linear(d_model, ssm_dim, bias=False)
        self.C = nn.Linear(ssm_dim, d_model, bias=False)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):  # x: [B, L, C]
        B, L, C = x.shape
        xn = self.norm(x)
        xc = self.dw_conv(xn.transpose(1,2)).transpose(1,2)  # [B, L, C]
        u = self.in_proj(xc)  # [B, L, S]
        h = torch.zeros(B, self.ssm_dim, device=x.device, dtype=x.dtype)
        outs = []
        A = torch.exp(self.A) * -1.0  # negative for stability
        for t in range(L):
            h = torch.exp(A) * h + self.B(xc[:, t, :])
            y = self.C(h)
            outs.append(y)
        y = torch.stack(outs, dim=1)  # [B, L, C]
        g = torch.sigmoid(self.gate(xn))
        return x + g * y

class M2CallerModel(nn.Module):
    def __init__(self, d_model=512, layers=2, heads=4, ssm_dim=256, vocab_size=5):
        super().__init__()
        self.sampler = SignalSampler(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1024, d_model) * 0.01)  # simple abs pos up to 1024 steps
        self.blocks = nn.ModuleList([Mamba2LikeBlock(d_model, ssm_dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, signal, signal_lengths):
        if signal.dim() == 3 and signal.shape[1] != 1:
            signal = signal.transpose(1,2)  # [B, 1, W]
        feat = self.sampler(signal)  # [B, C, L']
        B, C, Lp = feat.shape
        x = feat.transpose(1,2)  # [B, L', C]
        Lmax = min(self.pos.shape[1], Lp)
        x = x[:, :Lmax, :] + self.pos[:, :Lmax, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.out(x)  # [B, L', V]
        log_probs = logits.log_softmax(-1).transpose(0,1).contiguous()
        lengths = torch.full((B,), log_probs.shape[0], dtype=torch.int32, device=log_probs.device)
        return log_probs, lengths
