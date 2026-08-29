#!/usr/bin/env python3
"""Generate the full 50-isolate LaTeX results table for the benchmark paper.

Writes a two-column-spanning longtable-free table* with per-isolate median
identity for Bonito sup, Bonito hac, optional Dorado sup, and students.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'logs'
OUT = ROOT / 'bibm_benchmark' / 'latex' / 'full_table.tex'

SOURCES = [
    ('sup', LOG / 'bonito_sup_all.csv'),
    ('hac', LOG / 'bonito_hac_all.csv'),
    ('htiny', LOG / 'phase2' / 'eval50_p2_h_tiny.csv'),
    ('kdtiny', LOG / 'phase2' / 'eval50_p2_kdjp_tiny.csv'),
    ('hsmall', LOG / 'phase2' / 'eval50_p2_h_small.csv'),
    ('kdsmall', LOG / 'phase2' / 'eval50_p2_kdjp_small.csv'),
]
DORADO = ('dorado', LOG / 'dorado_sup_all.csv')


def load(path):
    return {r['isolate']: float(r['median']) * 100
            for r in csv.DictReader(open(path)) if r['isolate'] != 'OVERALL'}


def short(name):
    name = name.replace('Klebsiella_pneumoniae_', 'K.pn.\\ ')
    name = name.replace('Klebsiella_', 'K.\\ ')
    name = name.replace('_', '\\ ')
    return name[:30]


def main():
    data = {k: load(p) for k, p in SOURCES}
    has_dorado = DORADO[1].exists()
    if has_dorado:
        data[DORADO[0]] = load(DORADO[1])
    isos = sorted(data['sup'], key=lambda i: -data['sup'][i])

    if has_dorado:
        header = r'Isolate & sup & hac & Dorado & T-H & T-KD & S-H & S-KD \\'
        tabular = r'\begin{tabular}{lrrrrrrr}'
        caption_extra = r'; Dorado \texttt{sup} v3.6'
    else:
        header = r'Isolate & sup & hac & T-H & T-KD & S-H & S-KD \\'
        tabular = r'\begin{tabular}{lrrrrrr}'
        caption_extra = ''

    lines = [
        r'\begin{table*}[t]',
        r'\caption{Full per-isolate median read identity (\%) on NanoSpecies-50,'
        r' all 50 isolates, sorted by Bonito \texttt{sup}. T=Tiny (1.07\,M),'
        f' S=Small (3.51\\,M); H=hard-only, KD=CRF-KD{caption_extra}}}',
        r'\label{tab:full}',
        r'\centering',
        r'\scriptsize',
        r'\setlength{\tabcolsep}{4pt}',
        tabular,
        r'\toprule',
        header,
        r'\midrule',
    ]
    for i in isos:
        if has_dorado:
            row = (f'{short(i)} & {data["sup"][i]:.2f} & {data["hac"][i]:.2f} & '
                   f'{data["dorado"][i]:.2f} & {data["htiny"][i]:.2f} & '
                   f'{data["kdtiny"][i]:.2f} & {data["hsmall"][i]:.2f} & '
                   f'{data["kdsmall"][i]:.2f} \\\\')
        else:
            row = (f'{short(i)} & {data["sup"][i]:.2f} & {data["hac"][i]:.2f} & '
                   f'{data["htiny"][i]:.2f} & {data["kdtiny"][i]:.2f} & '
                   f'{data["hsmall"][i]:.2f} & {data["kdsmall"][i]:.2f} \\\\')
        lines.append(row)
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table*}',
    ]
    OUT.write_text('\n'.join(lines) + '\n')
    print('wrote', OUT, f'({len(isos)} isolates, dorado={has_dorado})')


if __name__ == '__main__':
    main()
