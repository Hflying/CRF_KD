"""Summarise Phase-2 ablation NanoSpecies-50 CSVs into icassppaper §3.4 Table 2.

Reads logs/phase2/eval50_abl_*.csv and writes:
  - logs/phase2/ablation_table.md
  - logs/phase2/ablation_table.tex
  - logs/phase2/ablation_table.tsv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'logs' / 'phase2'
LOG.mkdir(parents=True, exist_ok=True)


ROWS = [
    ('a1_hard',     'L_hard only'),
    ('a2_post',     '+ L_post'),
    ('a3_postfwd',  '+ L_post + L_fwd'),
    ('a4_filter',   '+ entropy filter'),
    ('a5_curric',   '+ curriculum'),
]


def load_per_iso(path: Path):
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['isolate'] == 'OVERALL':
                continue
            rows.append(float(r['median']))
    return rows


def main():
    summaries = []
    for tag, label in ROWS:
        path = LOG / f'eval50_abl_{tag}.csv'
        meds = load_per_iso(path)
        if meds is None:
            print(f'[skip] {tag}: missing {path}', file=sys.stderr)
            continue
        mom = sum(meds) / len(meds)
        summaries.append((label, mom))
        print(f'  {tag:12s}  iso={len(meds):2d}  mom={mom * 100:.2f} %')

    if not summaries:
        sys.exit('no ablation CSVs found')

    md = ['### Phase 2 loss ablation on NanoSpecies-50 (M2_SMALL, +10 ep over R2)', '']
    md.append('| Configuration | Median identity (mean of medians, %) | Δ vs row above |')
    md.append('|---|---:|---:|')
    prev = None
    for label, mom in summaries:
        cell = f'{mom * 100:.2f}'
        delta = '—' if prev is None else f'{(mom - prev) * 100:+.2f}'
        md.append(f'| {label} | {cell} | {delta} |')
        prev = mom
    (LOG / 'ablation_table.md').write_text('\n'.join(md) + '\n')

    tex = [r'\begin{tabular}{lrr}', r'\toprule',
           r'Configuration & Median (\%) & $\Delta$ vs above \\', r'\midrule']
    prev = None
    for label, mom in summaries:
        d = '---' if prev is None else f'{(mom - prev) * 100:+.2f}'
        tex.append(f'{label} & {mom * 100:.2f} & {d} \\\\')
        prev = mom
    tex += [r'\bottomrule', r'\end{tabular}']
    (LOG / 'ablation_table.tex').write_text('\n'.join(tex) + '\n')

    tsv = ['config\tmedian_pct\tdelta_pct']
    prev = None
    for label, mom in summaries:
        d = '' if prev is None else f'{(mom - prev) * 100:+.2f}'
        tsv.append(f'{label}\t{mom * 100:.2f}\t{d}')
        prev = mom
    (LOG / 'ablation_table.tsv').write_text('\n'.join(tsv) + '\n')

    print('\nWrote', LOG / 'ablation_table.md')


if __name__ == '__main__':
    main()
