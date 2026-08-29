#!/usr/bin/env python3
"""Generate figures for both BIBM papers from NanoSpecies CSVs."""
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# This script feeds both BIBM papers, which are released as separate
# repositories. Here the CRF-KD figures are the primary output; the NanoSpecies
# ones (see the companion repository) go to a subdirectory so that the two
# papers' method_overview.pdf do not overwrite each other.
FIG_KD = ROOT / 'figures'
FIG_BENCH = ROOT / 'figures' / 'companion_nanospecies'
for d in (FIG_KD, FIG_BENCH):
    d.mkdir(parents=True, exist_ok=True)


def load_isolate_medians(path):
    rows = [r for r in csv.DictReader(open(path)) if r['isolate'] != 'OVERALL']
    return [(r['isolate'], float(r['median'])) for r in rows]


def genus_of(isolate):
    return isolate.split('_')[0]


def fig_spread(out_path):
    data = load_isolate_medians(ROOT / 'logs/bonito_sup_all.csv')
    meds = [m * 100 for _, m in data]
    genera = sorted({genus_of(n) for n, _ in data})
    colors = plt.cm.tab20(np.linspace(0, 1, len(genera)))
    g2c = {g: colors[i] for i, g in enumerate(genera)}
    bar_colors = [g2c[genus_of(n)] for n, _ in data]
    order = sorted(range(len(data)), key=lambda i: data[i][1])
    labels = [data[i][0].replace('Klebsiella_pneumoniae_', 'K.pn.')[:22] for i in order]
    vals = [meds[i] for i in order]
    bc = [bar_colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(range(len(vals)), vals, color=bc, width=0.85, edgecolor='none')
    ax.axhline(statistics.median(meds), color='black', ls='--', lw=1, label='median 91.1%')
    ax.set_ylabel('Read identity (%)')
    ax.set_xlabel('Isolate (sorted by median)')
    ax.set_ylim(86, 96)
    ax.set_xticks([])
    ax.set_title('Bonito sup on NanoSpecies-50: 6.4 pp per-isolate spread')
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=g2c[g], label=g) for g in genera]
    ax.legend(handles=handles, ncol=4, fontsize=6, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_pareto(out_path):
    # params, median%, label
    pts = [
        (1.07, 85.03, 'M2-Tiny H'),
        (1.07, 85.27, 'M2-Tiny KD'),
        (3.51, 87.70, 'M2-Small H'),
        (3.51, 87.06, 'M2-Small KD'),
        (9.0, 89.98, 'Bonito hac'),
        (27.0, 91.08, 'Bonito sup'),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for p, m, lab in pts:
        mk = 's' if 'KD' in lab else 'o'
        col = '#2166ac' if 'M2' in lab else '#636363'
        ax.scatter(p, m, s=60, marker=mk, c=col, zorder=3)
        ax.annotate(lab, (p, m), textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('Parameters (M, log scale)')
    ax.set_ylabel('Median read identity (%)')
    ax.set_title('Accuracy vs. model size (NanoSpecies-50)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_ablation(out_path):
    configs = [
        ('Random', 25.0, '#cccccc'),
        ('H 4ep', 45.24, '#999999'),
        ('β-only KL', 28.09, '#d6604d'),
        ('β+L_fwd', 0.81, '#b2182b'),
        ('Joint KL λ=1', 72.67, '#2166ac'),
        ('H 8ep', 72.20, '#999999'),
        ('Joint 8ep', 77.62, '#2166ac'),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    x = range(len(configs))
    ax.bar(x, [c[1] for c in configs], color=[c[2] for c in configs])
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs], rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Median identity (%)')
    ax.set_ylim(0, 85)
    ax.set_title('Loss-formulation ablation (M2-Tiny, 1 iso)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def _load_med(path):
    rows = [r for r in csv.DictReader(open(path)) if r['isolate'] != 'OVERALL']
    return {r['isolate']: float(r['median']) * 100 for r in rows}


def fig_crossover(out_path):
    """Per-isolate KD-Hard delta for Tiny vs Small: the capacity crossover."""
    h_tiny = _load_med(ROOT / 'logs/phase2/eval50_p2_h_tiny.csv')
    kd_tiny = _load_med(ROOT / 'logs/phase2/eval50_p2_kdjp_tiny.csv')
    h_small = _load_med(ROOT / 'logs/phase2/eval50_p2_h_small.csv')
    kd_small = _load_med(ROOT / 'logs/phase2/eval50_p2_kdjp_small.csv')
    isos = sorted(h_tiny)
    d_tiny = np.array([kd_tiny[i] - h_tiny[i] for i in isos])
    d_small = np.array([kd_small[i] - h_small[i] for i in isos])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    parts = ax.violinplot([d_tiny, d_small], positions=[0, 1],
                          showmeans=True, showextrema=False, widths=0.7)
    for pc, col in zip(parts['bodies'], ['#2166ac', '#b2182b']):
        pc.set_facecolor(col)
        pc.set_alpha(0.5)
    parts['cmeans'].set_color('black')
    # jittered points
    rng = np.random.default_rng(1)
    for pos, d, col in [(0, d_tiny, '#2166ac'), (1, d_small, '#b2182b')]:
        jit = rng.normal(0, 0.04, size=len(d))
        ax.scatter(pos + jit, d, s=10, color=col, alpha=0.6, zorder=3)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['M2-Tiny\n(1.07 M)', 'M2-Small\n(3.51 M)'])
    ax.set_ylabel('Per-isolate $\\Delta$ identity\n(CRF-KD $-$ Hard, pp)')
    ax.set_title('Capacity-dependent KD crossover (n=50)')
    ax.annotate('+0.30 pp\n48/50 win\np=2e-9', (0, d_tiny.max()),
                textcoords='offset points', xytext=(8, -2), fontsize=6.5,
                color='#2166ac')
    ax.annotate('$-$0.66 pp\n0/50 win\np=8e-10', (1, d_small.min()),
                textcoords='offset points', xytext=(8, 4), fontsize=6.5,
                color='#b2182b')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_robustness(out_path):
    """Teacher vs student per-isolate identity: robustness compounding."""
    sup = _load_med(ROOT / 'logs/bonito_sup_all.csv')
    kd_tiny = _load_med(ROOT / 'logs/phase2/eval50_p2_kdjp_tiny.csv')
    h_small = _load_med(ROOT / 'logs/phase2/eval50_p2_h_small.csv')
    isos = sorted(sup)
    x = np.array([sup[i] for i in isos])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for st, lab, col, mk in [(h_small, 'M2-Small (3.51 M)', '#2166ac', 'o'),
                             (kd_tiny, 'M2-Tiny (1.07 M)', '#b2182b', '^')]:
        y = np.array([st[i] for i in isos])
        ax.scatter(x, y, s=18, color=col, alpha=0.7, marker=mk, label=lab)
    ax.set_xlabel('Teacher (Bonito sup) per-isolate identity (%)')
    ax.set_ylabel('Student per-isolate identity (%)')
    ax.set_title('Students fail where the teacher fails\n(Spearman $\\rho$=0.86--0.99)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_training_dynamics(out_path):
    """KD loss decomposition (posterior-KL vs hard term) for M2-Small KD run."""
    cols = ['epoch', 'step', 'total', 'post', 'hard', 'lr']

    def load_curve(path):
        ep, post, hard = [], [], []
        with open(path) as f:
            first = f.readline()
            f.seek(0)
            has_header = first.startswith('epoch')
            reader = csv.reader(f)
            if has_header:
                next(reader)
            for row in reader:
                if len(row) < 5:
                    continue
                ep.append(int(row[0]))
                post.append(float(row[3]))
                hard.append(float(row[4]))
        return np.array(ep), np.array(post), np.array(hard)

    def epoch_means(ep, vals):
        xs = sorted(set(ep.tolist()))
        return np.array(xs), np.array([vals[ep == e].mean() for e in xs])

    ek, pk, hk = load_curve(ROOT / 'checkpoints_kd/p2_kdjp_small/train.csv')
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xe, ph = epoch_means(ek, pk)
    _, hh = epoch_means(ek, hk)
    ax.plot(xe, hh, '-o', ms=3, color='#b2182b', label='hard term $\\mathcal{L}_\\mathrm{hard}$')
    ax.plot(xe, ph, '-s', ms=3, color='#2166ac', label='posterior-KL term $\\mathcal{L}_\\mathrm{post}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training loss (per-epoch mean)')
    ax.set_title('CRF-KD loss decomposition (M2-Small)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_ablation_scale(out_path):
    """β-only vs joint vs hard at validation and R1 scales."""
    scales = ['Val\n(1 iso)', 'R1\n(50 iso)']
    hard = [45.2, 76.4]
    beta = [28.1, 70.0]
    joint = [72.7, 78.9]
    x = np.arange(len(scales))
    w = 0.25
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(x - w, hard, w, label='Hard only', color='#636363')
    ax.bar(x, beta, w, label=r'$\beta$-only KL', color='#b2182b')
    ax.bar(x + w, joint, w, label='Joint KL', color='#2166ac')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    ax.set_ylabel('8-iso hold-out median (%)')
    ax.set_ylim(0, 90)
    ax.set_title(r'Posterior target ablation ($\beta$ vs joint)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_arch_control(out_path):
    """Architecture control: Hard vs CRF-KD delta for Mamba2 and CNN."""
    encoders = ['Mamba2\n(1.07 M)', 'CNN\n(1.00 M)']
    hard = [76.4, 73.3]
    kd = [78.9, 76.8]
    x = np.arange(2)
    w = 0.35
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.bar(x - w / 2, hard, w, label='Hard only', color='#636363')
    ax.bar(x + w / 2, kd, w, label='CRF-KD', color='#2166ac')
    for i, (h, k) in enumerate(zip(hard, kd)):
        ax.annotate(f'$\\Delta$={k - h:+.1f}', (i, k + 0.3), ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(encoders)
    ax.set_ylabel('8-iso hold-out median (%)')
    ax.set_ylim(70, 82)
    ax.set_title('Architecture control (R1 budget, M2-Tiny head)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_vendor_scatter(out_path):
    """Per-isolate identity: Bonito sup vs Dorado sup v3.6."""
    sup = _load_med(ROOT / 'logs/bonito_sup_all.csv')
    dor = _load_med(ROOT / 'logs/dorado_sup_all.csv')
    isos = sorted(sup)
    x = np.array([sup[i] for i in isos])
    y = np.array([dor[i] for i in isos])
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.scatter(x, y, s=22, color='#2166ac', alpha=0.75)
    lo, hi = min(x.min(), y.min()) - 0.5, max(x.max(), y.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Bonito sup v3.3 (%)')
    ax.set_ylabel('Dorado sup v3.6 (%)')
    ax.set_title('Vendor checkpoints track per-isolate\n(Spearman $\\rho$=0.99, spread preserved)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_corr_matrix(out_path):
    """Spearman rank-correlation of per-isolate identity across all 7 basecallers."""
    from scipy.stats import spearmanr
    files = [
        ('Bonito sup', 'logs/bonito_sup_all.csv'),
        ('Bonito hac', 'logs/bonito_hac_all.csv'),
        ('Dorado sup', 'logs/dorado_sup_all.csv'),
        ('M2-Tiny H', 'logs/phase2/eval50_p2_h_tiny.csv'),
        ('M2-Tiny KD', 'logs/phase2/eval50_p2_kdjp_tiny.csv'),
        ('M2-Small H', 'logs/phase2/eval50_p2_h_small.csv'),
        ('M2-Small KD', 'logs/phase2/eval50_p2_kdjp_small.csv'),
    ]
    d = {n: _load_med(ROOT / p) for n, p in files}
    isos = sorted(set.intersection(*[set(v) for v in d.values()]))
    names = [n for n, _ in files]
    mat = np.array([[d[a][i] for i in isos] for a in names])
    corr = np.array([[spearmanr(mat[i], mat[j]).correlation
                      for j in range(len(names))] for i in range(len(names))])

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(corr, cmap='YlGnBu', vmin=0.8, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=6.8)
    ax.set_yticklabels(names, fontsize=6.8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6.0, color='white' if v > 0.94 else '#1f2933')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Spearman $\\rho$ (per-isolate identity)', fontsize=7)
    ax.set_title('Isolate difficulty ordering is basecaller-invariant', fontsize=8, pad=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print('wrote', out_path)


def fig_method_overview(out_path):
    """Hybrid conceptual/data overview of the NanoSpecies-50 pipeline."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    sup = _load_med(ROOT / 'logs/bonito_sup_all.csv')
    hac = _load_med(ROOT / 'logs/bonito_hac_all.csv')
    dor = _load_med(ROOT / 'logs/dorado_sup_all.csv')

    fig = plt.figure(figsize=(7.2, 3.25))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.95, 1.05],
        left=0.035,
        right=0.985,
        bottom=0.105,
        top=0.965,
        hspace=0.38,
        wspace=0.32,
    )
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    edge = '#27364a'
    blue = '#2166ac'
    teal = '#72b7b2'
    orange = '#f58518'
    red = '#c43c4e'
    panel_fc = '#f8fbff'
    panels = [
        (0.02, 0.23, 0.15, 0.58, '50 isolates'),
        (0.23, 0.23, 0.15, 0.58, 'fixed reads'),
        (0.44, 0.23, 0.15, 0.58, 'checkpoints'),
        (0.65, 0.23, 0.15, 0.58, 'CRF decode'),
        (0.845, 0.23, 0.14, 0.58, 'metrics'),
    ]
    for x, y, w, h, title in panels:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.012,rounding_size=0.03',
            facecolor=panel_fc, edgecolor=edge, linewidth=0.9,
        ))
        ax.text(x + w / 2, y + h + 0.045, title, ha='center', va='bottom',
                fontsize=7.6, fontweight='bold', color=edge)

    # Corpus: colored isolate dots encode many genera rather than a text list.
    x, y, w, h, _ = panels[0]
    genera_cols = plt.cm.tab20(np.linspace(0, 1, 13))
    for j in range(50):
        cx = x + 0.025 + (j % 10) * 0.011
        cy = y + 0.18 + (j // 10) * 0.067
        ax.add_patch(Circle((cx, cy), 0.0046, color=genera_cols[j % 13], ec='none'))
    ax.text(x + w / 2, y + 0.075, '13 genera', ha='center', va='center',
            fontsize=6.4, color='#475467')

    # Manifest: deterministic selection as an ordered file strip with selected reads.
    x, y, w, h, _ = panels[1]
    for j in range(11):
        yy = y + 0.16 + j * 0.035
        ax.add_line(plt.Line2D([x + 0.028, x + w - 0.028], [yy, yy],
                               lw=1.0, color='#b7c7d9'))
        if j in (1, 3, 6, 9):
            ax.add_patch(Rectangle((x + 0.035, yy - 0.008), 0.028, 0.016,
                                   facecolor=blue, edgecolor='none', alpha=0.9))
    ax.text(x + w / 2, y + 0.075, '100 / isolate', ha='center', va='center',
            fontsize=6.4, color='#475467')

    # Checkpoints: model family chips instead of prose.
    x, y, w, h, _ = panels[2]
    chips = [('Bonito', blue), ('Dorado', teal), ('M2', orange)]
    for j, (lab, col) in enumerate(chips):
        yy = y + 0.46 - j * 0.12
        ax.add_patch(FancyBboxPatch((x + 0.025, yy), w - 0.05, 0.075,
                                    boxstyle='round,pad=0.007,rounding_size=0.018',
                                    facecolor=col, edgecolor='none', alpha=0.92))
        ax.text(x + w / 2, yy + 0.037, lab, ha='center', va='center',
                fontsize=6.7, color='white', fontweight='bold')

    # Decoder: small lattice with highlighted Viterbi path.
    x, y, w, h, _ = panels[3]
    nodes = []
    for col in range(4):
        for row in range(3):
            px = x + 0.035 + col * 0.029
            py = y + 0.21 + row * 0.09
            nodes.append((px, py))
            ax.add_patch(Circle((px, py), 0.006, facecolor='white',
                                edgecolor='#6b7280', linewidth=0.7))
    for col in range(3):
        for row in range(3):
            p0 = nodes[col * 3 + row]
            for dr in (-1, 0, 1):
                rr = row + dr
                if 0 <= rr < 3:
                    p1 = nodes[(col + 1) * 3 + rr]
                    ax.add_line(plt.Line2D([p0[0], p1[0]], [p0[1], p1[1]],
                                           lw=0.45, color='#cbd5e1', zorder=0))
    path = [nodes[1], nodes[5], nodes[7], nodes[10]]
    ax.add_line(plt.Line2D([p[0] for p in path], [p[1] for p in path],
                           lw=2.0, color=red, solid_capstyle='round'))
    ax.text(x + w / 2, y + 0.075, 'audited Viterbi', ha='center', va='center',
            fontsize=6.4, color='#475467')

    # Metrics: identity/spread as compact visual encodings.
    x, y, w, h, _ = panels[4]
    mini = [0.23, 0.31, 0.26]
    for j, val in enumerate(mini):
        bx = x + 0.035 + j * 0.024
        ax.add_patch(Rectangle((bx, y + 0.22), 0.016, val,
                               facecolor=[blue, teal, orange][j], edgecolor='none'))
    ax.add_line(plt.Line2D([x + 0.035, x + w - 0.035], [y + 0.50, y + 0.50],
                           color=red, lw=1.4))
    ax.text(x + w / 2, y + 0.075, 'median / spread', ha='center', va='center',
            fontsize=6.1, color='#475467')

    # A single routed bus keeps the concept flow visually controlled.
    for i in range(len(panels) - 1):
        x0 = panels[i][0] + panels[i][2] + 0.012
        x1 = panels[i + 1][0] - 0.012
        yy = panels[i][1] + panels[i][3] / 2
        ax.add_patch(FancyArrowPatch(
            (x0, yy), (x1, yy), arrowstyle='-|>',
            mutation_scale=10, linewidth=1.2, color=edge,
            shrinkA=0, shrinkB=0,
        ))

    ax.text(0.5, 0.05,
            'Deterministic pipeline: fixed reads + shared decoder -> comparable per-isolate evidence',
            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=edge)

    ax_spread = fig.add_subplot(gs[1, 0])
    isos = sorted(sup, key=sup.get)
    vals = np.array([sup[i] for i in isos])
    genera = sorted({genus_of(i) for i in isos})
    cmap = plt.cm.tab20(np.linspace(0, 1, len(genera)))
    g2c = {g: cmap[j] for j, g in enumerate(genera)}
    ax_spread.bar(np.arange(len(vals)), vals, color=[g2c[genus_of(i)] for i in isos],
                  width=0.92, edgecolor='none')
    ax_spread.axhline(np.median(vals), color='#1f2933', ls='--', lw=0.8)
    ax_spread.set_ylim(87, 95)
    ax_spread.set_xticks([])
    ax_spread.set_ylabel('Identity (%)', fontsize=6.5)
    ax_spread.set_title('50 isolates: 6.4 pp spread', fontsize=7.6, pad=3)
    ax_spread.tick_params(labelsize=6, length=2)
    ax_spread.grid(True, axis='y', alpha=0.25, lw=0.5)
    ax_spread.spines[['top', 'right']].set_visible(False)

    ax_vendor = fig.add_subplot(gs[1, 1])
    x = np.array([sup[i] for i in sorted(sup)])
    y = np.array([dor[i] for i in sorted(sup)])
    ax_vendor.scatter(x, y, s=10, color='#2166ac', alpha=0.72, edgecolors='none')
    lo, hi = 87.4, 94.8
    ax_vendor.plot([lo, hi], [lo, hi], color='#1f2933', ls='--', lw=0.8, alpha=0.55)
    ax_vendor.set_xlim(lo, hi)
    ax_vendor.set_ylim(lo, hi)
    ax_vendor.set_xlabel('Bonito sup', fontsize=6.5)
    ax_vendor.set_ylabel('Dorado sup', fontsize=6.5)
    ax_vendor.set_title('Vendor checkpoints preserve ranking', fontsize=7.6, pad=3)
    ax_vendor.text(0.04, 0.92, r'$\rho=0.99$', transform=ax_vendor.transAxes,
                   fontsize=7, fontweight='bold', color='#2166ac')
    ax_vendor.tick_params(labelsize=6, length=2)
    ax_vendor.grid(True, alpha=0.25, lw=0.5)
    ax_vendor.spines[['top', 'right']].set_visible(False)

    ax_models = fig.add_subplot(gs[1, 2])
    labels = ['Bonito\nsup', 'Dorado\nsup', 'Bonito\nhac', 'M2-S\nH', 'M2-T\nKD']
    medians = [
        statistics.median(sup.values()),
        statistics.median(dor.values()),
        statistics.median(hac.values()),
        87.70,
        85.27,
    ]
    bar_colors = ['#4c78a8', '#72b7b2', '#6b7280', '#f58518', '#e45756']
    ax_models.bar(np.arange(len(labels)), medians, color=bar_colors, width=0.72)
    ax_models.set_ylim(84, 92)
    ax_models.set_xticks(np.arange(len(labels)))
    ax_models.set_xticklabels(labels, fontsize=6.2)
    ax_models.set_ylabel('Median (%)', fontsize=6.5)
    ax_models.set_title('One protocol, multiple models', fontsize=7.6, pad=3)
    for j, v in enumerate(medians):
        ax_models.text(j, v + 0.15, f'{v:.1f}', ha='center', va='bottom',
                       fontsize=5.8, color='#1f2933')
    ax_models.tick_params(axis='y', labelsize=6, length=2)
    ax_models.grid(True, axis='y', alpha=0.25, lw=0.5)
    ax_models.spines[['top', 'right']].set_visible(False)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print('wrote', out_path)


def fig_lambda_sweep(out_path):
    lams = [0.25, 0.5, 1.0, 2.0, 4.0]
    meds = [77.49, 77.81, 78.21, 78.13, 77.81]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(lams, meds, 'o-', color='#2166ac', ms=6)
    ax.axvline(1.0, color='gray', ls=':', lw=1)
    ax.set_xscale('log', base=2)
    ax.set_xticks(lams)
    ax.set_xticklabels([str(v) for v in lams])
    ax.set_xlabel(r'$\lambda_{\mathrm{post}}$ ($\lambda_{\mathrm{hard}}=1$)')
    ax.set_ylabel('Validation median (%)')
    ax.set_ylim(77, 78.5)
    ax.set_title(r'$\lambda_{\mathrm{post}}$ sensitivity (M2-Tiny, 1 iso)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


def fig_kd_method_overview(out_path):
    """Hybrid conceptual/data overview of CRF-KD."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.04, 1.00],
        left=0.04,
        right=0.985,
        bottom=0.11,
        top=0.94,
        hspace=0.48,
        wspace=0.35,
    )

    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    edge = '#334155'
    # Signal icon: a small current trace instead of another text box.
    sig = FancyBboxPatch((0.035, 0.28), 0.16, 0.46,
                         boxstyle='round,pad=0.012,rounding_size=0.035',
                         facecolor='#f8fafc', edgecolor=edge, linewidth=0.9)
    ax.add_patch(sig)
    xs = np.linspace(0.055, 0.175, 120)
    ys = 0.51 + 0.07 * np.sin(np.linspace(0, 4 * np.pi, 120)) + 0.018 * np.sin(np.linspace(0, 17 * np.pi, 120))
    ax.plot(xs, ys, color='#2563eb', lw=1.5)
    ax.text(0.115, 0.34, 'raw signal', ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#0f172a')

    def lane_box(x, y, title, subtitle, fc, ec=edge):
        box = FancyBboxPatch((x, y), 0.16, 0.22,
                             boxstyle='round,pad=0.014,rounding_size=0.035',
                             facecolor=fc, edgecolor=ec, linewidth=0.9)
        ax.add_patch(box)
        ax.text(x + 0.08, y + 0.145, title, ha='center', va='center',
                fontsize=8.0, fontweight='bold', color='#0f172a')
        ax.text(x + 0.08, y + 0.065, subtitle, ha='center', va='center',
                fontsize=6.6, color='#334155')

    lane_box(0.265, 0.54, 'Teacher', r'Bonito sup, frozen', '#dbeafe')
    lane_box(0.265, 0.28, 'Student', r'Mamba2 / CNN', '#fed7aa')

    # Central lattice/posterior visual: heatmap cells plus path edges.
    lattice = FancyBboxPatch((0.475, 0.24), 0.18, 0.50,
                             boxstyle='round,pad=0.014,rounding_size=0.035',
                             facecolor='#eff6ff', edgecolor='#1d4ed8', linewidth=1.0)
    ax.add_patch(lattice)
    rng = np.random.default_rng(7)
    for i in range(4):
        for j in range(3):
            val = 0.35 + 0.45 * rng.random()
            ax.add_patch(Rectangle((0.502 + 0.033 * i, 0.34 + 0.067 * j),
                                   0.025, 0.045, facecolor=plt.cm.Blues(val),
                                   edgecolor='white', linewidth=0.4))
    for x0, y0, x1, y1 in [(0.505, 0.36, 0.54, 0.43), (0.54, 0.43, 0.575, 0.50),
                           (0.575, 0.50, 0.61, 0.43), (0.61, 0.43, 0.625, 0.57)]:
        ax.plot([x0, x1], [y0, y1], color='#1d4ed8', lw=1.2, alpha=0.9)
    ax.text(0.565, 0.690, r'joint posterior', ha='center', va='center',
            fontsize=7.8, fontweight='bold', color='#1e3a8a')
    ax.text(0.565, 0.615, r'$P_t^*(s,x)$', ha='center', va='center',
            fontsize=8.6, color='#1e40af')
    ax.text(0.565, 0.285, r'normalize over $(s,x)$', ha='center', va='center',
            fontsize=6.6, color='#1e3a8a')

    # Loss node: two incoming signals, one from posterior matching and one from Viterbi targets.
    loss = FancyBboxPatch((0.775, 0.34), 0.18, 0.34,
                          boxstyle='round,pad=0.014,rounding_size=0.035',
                          facecolor='#fff1f2', edgecolor='#be123c', linewidth=0.95)
    ax.add_patch(loss)
    ax.text(0.865, 0.58, 'CRF-KD loss', ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#881337')
    ax.text(0.865, 0.48, r'$\lambda_{\rm post}$ KL', ha='center', va='center',
            fontsize=7.2, color='#881337')
    ax.text(0.865, 0.39, r'$+\ \lambda_{\rm hard}$ CTC-CRF', ha='center', va='center',
            fontsize=7.2, color='#881337')

    # Clean routed arrows: all model/posterior connections are horizontal.
    # The only split is a short input bus with clear spacing from boxes.
    arrow_kw = dict(arrowstyle='-|>', mutation_scale=10, linewidth=1.05)
    ax.add_patch(FancyArrowPatch((0.195, 0.51), (0.225, 0.51),
                                 color=edge, **arrow_kw))
    ax.plot([0.225, 0.225], [0.39, 0.65], color=edge, lw=1.0)
    ax.add_patch(FancyArrowPatch((0.225, 0.65), (0.265, 0.65),
                                 color=edge, **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.225, 0.39), (0.265, 0.39),
                                 color=edge, **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.425, 0.65), (0.475, 0.65),
                                 color='#1d4ed8', **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.425, 0.39), (0.475, 0.39),
                                 color='#f97316', **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.655, 0.51), (0.775, 0.51),
                                 color='#7c3aed', linewidth=1.25,
                                 arrowstyle='-|>', mutation_scale=11))
    ax.text(0.715, 0.55, 'supervision', ha='center', va='bottom',
            fontsize=6.4, fontweight='bold', color='#6d28d9')

    # Compact callout: visual note, not a large text-heavy box.
    for x in [0.515, 0.565, 0.615]:
        ax.add_patch(Circle((x, 0.13), 0.014, facecolor='#c2410c', edgecolor='none'))
    ax.text(0.565, 0.13, r'joint lattice target $\ne$ $\beta$-only API',
            ha='center', va='center', fontsize=7.4, fontweight='bold',
            color='#9a3412',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#fff7ed',
                      edgecolor='#c2410c', linewidth=0.8))

    ax_ab = fig.add_subplot(gs[1, 0])
    labels = [r'$\beta$-only', 'Hard', 'Joint']
    vals = [70.0, 76.4, 78.9]
    cols = ['#b2182b', '#737373', '#2166ac']
    ax_ab.bar(range(3), vals, color=cols, width=0.68)
    ax_ab.set_ylim(68, 80.5)
    ax_ab.set_xticks(range(3))
    ax_ab.set_xticklabels(labels, fontsize=6.7)
    ax_ab.set_ylabel('8-iso median (%)', fontsize=6.5)
    ax_ab.set_title('Posterior target matters', fontsize=7.6, pad=3)
    for i, v in enumerate(vals):
        ax_ab.text(i, v + 0.25, f'{v:.1f}', ha='center', fontsize=6.1)
    ax_ab.grid(True, axis='y', alpha=0.25, lw=0.5)
    ax_ab.spines[['top', 'right']].set_visible(False)
    ax_ab.tick_params(axis='y', labelsize=6, length=2)

    ax_cross = fig.add_subplot(gs[1, 1])
    x = np.arange(2)
    means = [0.30, -0.66]
    cis = [[0.05, 0.10], [0.05, 0.10]]
    ax_cross.bar(x, means, color=['#2166ac', '#b2182b'], width=0.62)
    ax_cross.errorbar(x, means, yerr=np.array(cis).T, fmt='none',
                      ecolor='#111827', capsize=3, lw=0.8)
    ax_cross.axhline(0, color='#374151', lw=0.8, ls='--')
    ax_cross.set_xticks(x)
    ax_cross.set_xticklabels(['Tiny\n1.07M', 'Small\n3.51M'], fontsize=6.7)
    ax_cross.set_ylabel(r'KD $-$ Hard (pp)', fontsize=6.5)
    ax_cross.set_title('Capacity crossover (n=50)', fontsize=7.6, pad=5)
    ax_cross.text(0, 0.13, '48/50', ha='center', va='center',
                  fontsize=6.4, color='white', fontweight='bold')
    ax_cross.text(1, -0.37, '0/50', ha='center', va='center',
                  fontsize=6.4, color='white', fontweight='bold')
    ax_cross.set_ylim(-1.05, 0.65)
    ax_cross.grid(True, axis='y', alpha=0.25, lw=0.5)
    ax_cross.spines[['top', 'right']].set_visible(False)
    ax_cross.tick_params(axis='y', labelsize=6, length=2)

    ax_front = fig.add_subplot(gs[1, 2])
    pts = [
        (1.07, 85.03, 'Tiny H', '#94a3b8', 'o'),
        (1.07, 85.27, 'Tiny KD', '#2166ac', 's'),
        (3.51, 87.70, 'Small H', '#111827', 'o'),
        (3.51, 87.06, 'Small KD', '#b2182b', 's'),
        (27.0, 91.08, 'Teacher', '#525252', 'o'),
    ]
    for p, m, lab, col, mk in pts:
        ax_front.scatter(p, m, s=32, color=col, marker=mk, zorder=3)
    ax_front.annotate('+0.30 pp', (1.07, 85.27), xytext=(8, 7),
                      textcoords='offset points', fontsize=6.1, color='#2166ac')
    ax_front.annotate('-0.66 pp', (3.51, 87.06), xytext=(6, -11),
                      textcoords='offset points', fontsize=6.1, color='#b2182b')
    ax_front.set_xscale('log')
    ax_front.set_xlim(0.75, 35)
    ax_front.set_ylim(84.6, 91.5)
    ax_front.set_xlabel('Params (M, log)', fontsize=6.5)
    ax_front.set_ylabel('Median identity (%)', fontsize=6.5)
    ax_front.set_title('Deployable frontier', fontsize=7.6, pad=3)
    ax_front.grid(True, alpha=0.25, lw=0.5)
    ax_front.spines[['top', 'right']].set_visible(False)
    ax_front.tick_params(labelsize=6, length=2)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print('wrote', out_path)


def fig_delta50(out_path):
    """Per-isolate P2 delta (CRF-KD - Hard) for Tiny and Small on all 50 isolates."""
    h_tiny = _load_med(ROOT / 'logs/phase2/eval50_p2_h_tiny.csv')
    kd_tiny = _load_med(ROOT / 'logs/phase2/eval50_p2_kdjp_tiny.csv')
    h_small = _load_med(ROOT / 'logs/phase2/eval50_p2_h_small.csv')
    kd_small = _load_med(ROOT / 'logs/phase2/eval50_p2_kdjp_small.csv')
    isos = sorted(h_tiny, key=lambda i: kd_tiny[i] - h_tiny[i], reverse=True)
    d_tiny = np.array([kd_tiny[i] - h_tiny[i] for i in isos])
    d_small = np.array([kd_small[i] - h_small[i] for i in isos])

    def short(n):
        return n.replace('Klebsiella_pneumoniae_', 'K.pn.').replace('_', ' ')[:24]

    labels = [short(i) for i in isos]
    y = np.arange(len(isos))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharey=True)
    for ax, d, title, col_pos, col_neg in [
        (axes[0], d_tiny, 'M2-Tiny (1.07 M)', '#2166ac', '#b2182b'),
        (axes[1], d_small, 'M2-Small (3.51 M)', '#2166ac', '#b2182b'),
    ]:
        colors = [col_pos if v >= 0 else col_neg for v in d]
        ax.barh(y, d, color=colors, height=0.82)
        ax.axvline(0, color='gray', lw=0.8)
        ax.set_xlabel('$\\Delta$ identity (CRF-KD $-$ Hard, pp)')
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-1.85, 0.85)
        ax.grid(True, axis='x', alpha=0.25)
        ax.invert_yaxis()
    axes[0].set_yticks(y[::5])
    axes[0].set_yticklabels([labels[i] for i in range(0, len(labels), 5)], fontsize=5)
    fig.suptitle('P2 per-isolate $\\Delta$ on NanoSpecies-50 (sorted by Tiny $\\Delta$)',
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('wrote', out_path)


if __name__ == '__main__':
    fig_spread(FIG_BENCH / 'spread.pdf')
    fig_spread(FIG_KD / 'spread.pdf')
    fig_pareto(FIG_KD / 'pareto.pdf')
    fig_ablation(FIG_KD / 'ablation.pdf')
    fig_ablation_scale(FIG_KD / 'ablation_scale.pdf')
    fig_crossover(FIG_KD / 'crossover.pdf')
    fig_training_dynamics(FIG_KD / 'training.pdf')
    fig_robustness(FIG_BENCH / 'robustness.pdf')
    fig_arch_control(FIG_KD / 'arch_control.pdf')
    fig_vendor_scatter(FIG_BENCH / 'vendor_scatter.pdf')
    fig_corr_matrix(FIG_BENCH / 'corr_matrix.pdf')
    fig_method_overview(FIG_BENCH / 'method_overview.pdf')
    fig_lambda_sweep(FIG_KD / 'lambda.pdf')
    fig_kd_method_overview(FIG_KD / 'method_overview.pdf')
    fig_delta50(FIG_KD / 'delta50.pdf')
