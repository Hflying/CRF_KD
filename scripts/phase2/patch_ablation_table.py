#!/usr/bin/env python3
"""Patch tab:ablation_scale R1 β-only cell from r1_betaonly/eval.txt."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'checkpoints_kd' / 'r1_betaonly' / 'eval.txt'
TEX = ROOT / 'bibm_kd' / 'latex' / 'paper.tex'

def main():
    if not EVAL.exists():
        print(f'missing {EVAL}')
        return 1
    m = re.search(r'overall_median\s+([\d.]+)', EVAL.read_text())
    if not m:
        print('no overall_median in eval.txt')
        return 1
    val = float(m.group(1)) * 100
    text = TEX.read_text()
    new = re.sub(
        r'(50 iso, 3 ep \(R1\) & 76\.4 & )([^&]+)( & \\textbf\{78\.9\})',
        rf'\g<1>{val:.1f}\g<3>',
        text,
        count=1,
    )
    if new == text:
        print('pattern not found — already patched?')
        return 1
    TEX.write_text(new)
    print(f'patched R1 β-only = {val:.1f}%')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
