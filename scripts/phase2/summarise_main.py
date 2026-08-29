"""Summarise the Phase 2 main NanoSpecies-50 results into one Markdown
table that drops directly into icassppaper/paper.md §3.3 (Table 1).

Reads the four ``logs/phase2/eval50_*.csv`` student CSVs plus the existing
``logs/bonito_sup_all.csv`` teacher CSV, computes mean-of-medians /
median-of-medians per row, and writes:

  - logs/phase2/main_table.md   Markdown table (paper drop-in)
  - logs/phase2/main_table.tex  booktabs LaTeX
  - logs/phase2/main_table.tsv  Raw TSV
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'logs' / 'phase2'
LOG.mkdir(parents=True, exist_ok=True)


# (tag, paper-row-label, source-CSV)
ROWS = [
    ('sup',      'Bonito sup (teacher)',  ROOT / 'logs' / 'bonito_sup_all.csv'),
    ('h_tiny',   'M2_TINY  (hard-only)',  LOG  / 'eval50_p2_h_tiny.csv'),
    ('kd_tiny',  'M2_TINY  (CRF-KD ours)',LOG  / 'eval50_p2_kdjp_tiny.csv'),
    ('h_small',  'M2_SMALL (hard-only)',  LOG  / 'eval50_p2_h_small.csv'),
    ('kd_small', 'M2_SMALL (CRF-KD ours)',LOG  / 'eval50_p2_kdjp_small.csv'),
]


def load_per_iso(path: Path):
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['isolate'] == 'OVERALL':
                continue
            rows.append((r['isolate'], int(r['n_reads']),
                         float(r['median']), float(r['mean'])))
    return rows


def stats(rows):
    medians = [r[2] for r in rows]
    means = [r[3] for r in rows]
    n = len(rows)
    mom = sum(medians) / n  # mean of medians
    medmed = sorted(medians)[n // 2]
    momn = sum(means) / n
    return {
        'n_iso': n,
        'mean_of_medians': mom,
        'median_of_medians': medmed,
        'mean_of_means': momn,
    }


def fmt_pct(x):
    return f'{x * 100:.2f}'


def main():
    print('Reading per-CSV...')
    summaries = {}
    for tag, label, path in ROWS:
        per_iso = load_per_iso(path)
        if per_iso is None:
            print(f'  [skip] {tag}: missing {path}', file=sys.stderr)
            continue
        summaries[tag] = (label, stats(per_iso), len(per_iso))
        print(f'  {tag:10s}  iso={len(per_iso):2d}  '
              f'mom={summaries[tag][1]["mean_of_medians"]:.4f}')

    if 'sup' not in summaries:
        sys.exit('Bonito sup CSV missing; cannot compute Δ-to-teacher')
    sup_mom = summaries['sup'][1]['mean_of_medians']

    md_lines = [
        '### Phase 2 main result on NanoSpecies-50',
        '',
        '| Model | n_iso | Median (%) | Mean (%) | Δ to teacher (pp) |',
        '|---|---:|---:|---:|---:|',
    ]
    tex_lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        r'Model & $n_\text{iso}$ & Median (\%) & Mean (\%) & $\Delta$ to teacher (pp) \\',
        r'\midrule',
    ]
    tsv_lines = ['model\tn_iso\tmedian\tmean\tdelta_to_teacher_pp']

    order = ['sup', 'h_tiny', 'kd_tiny', 'h_small', 'kd_small']
    for tag in order:
        if tag not in summaries:
            continue
        label, st, n_iso = summaries[tag]
        med = st['mean_of_medians']
        mean = st['mean_of_means']
        delta = (med - sup_mom) * 100
        delta_str = '—' if tag == 'sup' else f'{delta:+.2f}'
        md_lines.append(
            f'| {label} | {n_iso} | {fmt_pct(med)} | {fmt_pct(mean)} | {delta_str} |'
        )
        tex_lines.append(
            f'{label} & {n_iso} & {fmt_pct(med)} & {fmt_pct(mean)} & '
            f'{delta_str.replace("—", "---")} \\\\'
        )
        tsv_lines.append(
            f'{label}\t{n_iso}\t{fmt_pct(med)}\t{fmt_pct(mean)}\t{delta_str}'
        )

    tex_lines += [r'\bottomrule', r'\end{tabular}']
    md_lines.append('')
    md_lines.append(
        '**Headline:** mean-of-medians of M2_TINY (CRF-KD) = '
        + (fmt_pct(summaries["kd_tiny"][1]["mean_of_medians"]) + ' %' if "kd_tiny" in summaries else "—")
        + ', M2_SMALL (CRF-KD) = '
        + (fmt_pct(summaries["kd_small"][1]["mean_of_medians"]) + ' %' if "kd_small" in summaries else "—")
        + ', against Bonito sup ' + fmt_pct(sup_mom) + ' %.'
    )

    (LOG / 'main_table.md').write_text('\n'.join(md_lines) + '\n')
    (LOG / 'main_table.tex').write_text('\n'.join(tex_lines) + '\n')
    (LOG / 'main_table.tsv').write_text('\n'.join(tsv_lines) + '\n')
    print('\nWrote', LOG / 'main_table.md')
    print('       ', LOG / 'main_table.tex')
    print('       ', LOG / 'main_table.tsv')


if __name__ == '__main__':
    main()
