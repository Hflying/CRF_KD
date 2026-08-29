#!/usr/bin/env python3
"""Compute real statistics for both BIBM papers from per-isolate CSVs.

Outputs paired tests (Wilcoxon), bootstrap CIs over isolates for KD-vs-Hard
deltas, and non-Klebsiella subset spreads. All numbers are paper-ready.
"""
import csv
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'logs' / 'phase2'
RNG = np.random.default_rng(0)


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r['isolate'] != 'OVERALL']
    return {r['isolate']: float(r['median']) * 100 for r in rows}


def genus(iso):
    return iso.split('_')[0]


def boot_ci(diffs, n=20000, alpha=0.05):
    diffs = np.asarray(diffs)
    means = np.array([RNG.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


def paired_report(name, hard, kd):
    isos = sorted(hard)
    h = np.array([hard[i] for i in isos])
    k = np.array([kd[i] for i in isos])
    d = k - h  # positive = KD better
    mean_d, lo, hi = boot_ci(d)
    try:
        w_stat, p_w = stats.wilcoxon(k, h)
    except ValueError:
        w_stat, p_w = float('nan'), float('nan')
    t_stat, p_t = stats.ttest_rel(k, h)
    wins = int((d > 0).sum())
    print(f'\n=== {name} (KD - Hard), n={len(isos)} isolates ===')
    print(f'  mean delta      : {mean_d:+.3f} pp')
    print(f'  bootstrap 95% CI: [{lo:+.3f}, {hi:+.3f}] pp')
    print(f'  KD wins / total : {wins}/{len(isos)}')
    print(f'  Wilcoxon signed : W={w_stat:.1f}  p={p_w:.4g}')
    print(f'  paired t-test   : t={t_stat:.3f}  p={p_t:.4g}')
    sig = 'SIGNIFICANT' if p_w < 0.05 else 'n.s.'
    direction = 'KD>Hard' if mean_d > 0 else 'Hard>KD'
    print(f'  -> {direction}, {sig}')
    return dict(mean=mean_d, lo=lo, hi=hi, p_w=p_w, wins=wins, n=len(isos))


def tail_breakdown(name, hard, kd, tail_genera):
    print(f'\n--- {name} tail-genus per-isolate (KD - Hard) ---')
    for i in sorted(hard):
        if genus(i) in tail_genera:
            print(f'  {i:42s} H={hard[i]:.2f} KD={kd[i]:.2f} d={kd[i]-hard[i]:+.2f}')


def spread_stats(name, medians, exclude_genus=None):
    vals = [v for i, v in medians.items()
            if exclude_genus is None or genus(i) != exclude_genus]
    vals = np.array(vals)
    print(f'\n### spread {name}: n={len(vals)} '
          f'median={np.median(vals):.2f} '
          f'min={vals.min():.2f} max={vals.max():.2f} '
          f'spread={vals.max()-vals.min():.2f} '
          f'IQR={np.percentile(vals,75)-np.percentile(vals,25):.2f} '
          f'std={vals.std(ddof=1):.2f}')
    return vals


sup = load(ROOT / 'logs' / 'bonito_sup_all.csv')
h_tiny = load(LOG / 'eval50_p2_h_tiny.csv')
kd_tiny = load(LOG / 'eval50_p2_kdjp_tiny.csv')
h_small = load(LOG / 'eval50_p2_h_small.csv')
kd_small = load(LOG / 'eval50_p2_kdjp_small.csv')

print('=' * 60)
print('PAPER 1 (CRF-KD): paired significance over 50 isolates')
print('=' * 60)
r_tiny = paired_report('M2-Tiny', h_tiny, kd_tiny)
r_small = paired_report('M2-Small', h_small, kd_small)

tail = {'Moraxella', 'Haemophilus', 'Acinetobacter', 'Comamonas'}
tail_breakdown('M2-Tiny', h_tiny, kd_tiny, tail)
tail_breakdown('M2-Small', h_small, kd_small, tail)

# Tail vs head split for Tiny
print('\n--- M2-Tiny: tail vs head mean delta ---')
isos = sorted(h_tiny)
tail_d = [kd_tiny[i] - h_tiny[i] for i in isos if genus(i) in tail]
head_d = [kd_tiny[i] - h_tiny[i] for i in isos if genus(i) not in tail]
print(f'  tail ({len(tail_d)} iso) mean d = {np.mean(tail_d):+.3f}')
print(f'  head ({len(head_d)} iso) mean d = {np.mean(head_d):+.3f}')

print('\n' + '=' * 60)
print('PAPER 2 (Benchmark): spread incl/excl Klebsiella')
print('=' * 60)
spread_stats('Bonito sup ALL', sup)
spread_stats('Bonito sup non-Klebsiella', sup, exclude_genus='Klebsiella')
# intra-Klebsiella pneumoniae
kp = {i: v for i, v in sup.items() if i.startswith('Klebsiella_pneumoniae')}
spread_stats('Bonito sup K.pneumoniae only', kp)

# Teacher-student correlation: does student degrade where teacher does?
print('\n### teacher-student rank correlation (robustness compounding) ###')
isos = sorted(sup)
for name, st in [('Tiny-KD', kd_tiny), ('Small-Hard', h_small)]:
    s_t = np.array([sup[i] for i in isos])
    s_s = np.array([st[i] for i in isos])
    rho, p = stats.spearmanr(s_t, s_s)
    print(f'  sup vs {name}: Spearman rho={rho:.3f} p={p:.3g}')
