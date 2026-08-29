#!/usr/bin/env python3
"""Summarise the lambda/seed sweep into paper-ready numbers and a figure."""
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/ubuntu/hardy/M2Caller')
CKPT = ROOT / 'checkpoints_kd'
FIG_KD = ROOT / 'bibm_kd' / 'figures'


def read_eval(name):
    p = CKPT / f'sweep_{name}' / 'eval.txt'
    if not p.exists():
        return None
    m = re.search(r'overall_median\s+([\d.]+)', p.read_text())
    return float(m.group(1)) * 100 if m else None


print('=== lambda_post sweep (KD, joint, M2-Tiny, INF078, 4ep) ===')
lams = [0.25, 0.5, 1.0, 2.0, 4.0]
lam_vals = []
for lp in lams:
    v = read_eval(f'lam{lp}')
    lam_vals.append(v)
    print(f'  lambda_post={lp:<5}  median={v if v is None else f"{v:.2f}"}%')

print('\n=== seed repeats ===')
for branch in ['H', 'KD']:
    vals = [read_eval(f'{branch}_seed{s}') for s in (42, 1, 7)]
    vals = [v for v in vals if v is not None]
    if len(vals) >= 2:
        print(f'  {branch}: {[f"{v:.2f}" for v in vals]}  '
              f'mean={statistics.mean(vals):.2f}  std={statistics.stdev(vals):.3f}')
    elif vals:
        print(f'  {branch}: {[f"{v:.2f}" for v in vals]} (incomplete)')

# Figure: lambda sensitivity curve
have = [(lp, v) for lp, v in zip(lams, lam_vals) if v is not None]
if len(have) >= 2:
    xs = [p[0] for p in have]
    ys = [p[1] for p in have]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(xs, ys, '-o', color='#2166ac')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda_{\mathrm{post}}$ (log scale, $\lambda_{\mathrm{hard}}=1$)')
    ax.set_ylabel('Median identity (%)')
    ax.set_title(r'CRF-KD $\lambda_{\mathrm{post}}$ sensitivity (M2-Tiny, val)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_KD / 'lambda.pdf'
    fig.savefig(out, dpi=200)
    print(f'\nwrote {out}')
else:
    print('\n[lambda figure skipped: <2 points]')
