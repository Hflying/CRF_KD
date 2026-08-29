#!/usr/bin/env python3
"""Patch benchmark paper tab:multi with Dorado sup v3.6 median."""
import csv
import re
from pathlib import Path

ROOT = Path('/home/ubuntu/hardy/M2Caller')
CSV = ROOT / 'logs' / 'dorado_sup_all.csv'
TEX = ROOT / 'bibm_benchmark' / 'latex' / 'paper.tex'
SUP = 91.08


def median_of_medians(path):
    meds = [float(r['median']) * 100
            for r in csv.DictReader(open(path)) if r['isolate'] != 'OVERALL']
    return float(__import__('numpy').median(meds))


def main():
    if not CSV.exists():
        print('missing', CSV)
        return 1
    med = median_of_medians(CSV)
    delta = med - SUP
    text = TEX.read_text()
    row = (f'Dorado \\texttt{{sup}} v3.6 & 27\\,M & {med:.2f}\\,\\% & '
           f'${delta:+.2f}$ \\\\')
    new = re.sub(
        r'Dorado \\texttt\{sup\} v3\.6 & 27\\,M & [\d.]+\\,\\% & \$[^\$]+\$ \\\\|'
        r'Dorado \\texttt\{sup\} v3\.6 & 27\\,M & TBD & TBD \\\\',
        row,
        text,
        count=1,
    )
    if new == text:
        # insert after Bonito hac row if placeholder missing
        needle = (
            'Bonito \\texttt{hac} & 9\\,M & 89.98\\,\\% & $-1.10$ \\\\\n'
            '\\midrule'
        )
        insert = (
            'Bonito \\texttt{hac} & 9\\,M & 89.98\\,\\% & $-1.10$ \\\\\n'
            f'{row}\n'
            '\\midrule'
        )
        if needle in text:
            new = text.replace(needle, insert, 1)
        else:
            print('insert point not found')
            return 1
    TEX.write_text(new)
    print(f'patched Dorado sup: {med:.2f}% delta={delta:+.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
