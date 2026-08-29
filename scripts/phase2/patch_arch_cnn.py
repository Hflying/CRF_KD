#!/usr/bin/env python3
"""Patch KD paper with CNN architecture-control results."""
import re
from pathlib import Path

ROOT = Path('/home/ubuntu/hardy/M2Caller')
TEX = ROOT / 'bibm_kd' / 'latex' / 'paper.tex'


def read_med(tag):
    p = ROOT / 'checkpoints_kd' / f'arch_cnn_{tag}' / 'eval.txt'
    if not p.exists():
        return None
    m = re.search(r'overall_median\s+([\d.]+)', p.read_text())
    return float(m.group(1)) * 100 if m else None


def main():
    h, kd = read_med('h'), read_med('kd')
    if h is None or kd is None:
        print(f'incomplete: h={h} kd={kd}')
        return 1
    delta = kd - h
    text = TEX.read_text()
    # table row
    new = re.sub(
        r'(CNN \(1\.00\\,M\) & )([\d.]+|TBD)( & )([\d.]+|TBD)( & )([\d.+-]+|TBD)( \\\\)',
        rf'\g<1>{h:.1f}\g<3>{kd:.1f}\g<5>{delta:+.1f}\g<7>',
        text,
        count=1,
    )
    if new == text:
        print('table pattern not found')
        return 1
    TEX.write_text(new)
    print(f'patched CNN arch: H={h:.1f} KD={kd:.1f} delta={delta:+.1f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
