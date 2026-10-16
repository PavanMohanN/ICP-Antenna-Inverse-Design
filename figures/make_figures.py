#!/usr/bin/env python3
"""
make_figures.py  -  regenerates every machine-learning figure of the EuCAP 2027 paper
"Inverse, Constrained and Physically Consistent Learning for the Design of a Low-Profile Wideband Patch Antenna".

    Fig. 2  fig_framework   ICP framework block diagram                      (double column)
    Fig. 3  fig_extraction  frequency hold-out test of the equivalent circuit (single column)
    Fig. 4  fig_forward     PC-Net on unseen geometries, panels (a)-(c)      (double column)
    Fig. 5  fig_inverse     inverse design, panels (a)-(c)                   (double column)
    (Fig. 1, the antenna drawing, is the authors' own image Pics/Ant.png.)

Usage (from the repository root; `python run.py figures` does the same)
    python figures/make_figures.py                              # paper run -> figures/output (PDF + 300 dpi PNG)
    python figures/make_figures.py --only forward inverse       # selected figures
    python figures/make_figures.py --data runs/quick/figure_data --out runs/quick/figures
    python figures/make_figures.py --out paper/figures --formats pdf   # overwrite the figures used by Paper.tex

Requirements: numpy and matplotlib only. All data are read from a figure_data folder (default: the paper run,
runs/paper/figure_data, written by pipeline step 15), so nothing has to be retrained to restyle a figure.

How to edit
    * Global look (fonts, sizes, line widths, colours)  -> STYLE and COLOURS below.
    * Figure size                                       -> the WIDTH/HEIGHT constants in each function.
    * Labels, legends, panel letters, axis limits       -> inside each fig_* function (commented per panel).
    * Paper file names are kept, so the output can be copied straight into paper/figures/.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'runs', 'paper', 'figure_data')   # replaced by --data

# ============================== STYLE ==============================
COL1, COL2 = 3.5, 7.16          # IEEE single- and double-column widths (inch)
STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'TeX Gyre Termes', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'lines.linewidth': 1.2, 'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.4,
    'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,          # embed TrueType fonts (IEEE PDF eXpress compliant)
}
plt.rcParams.update(STYLE)

# Colour-blind-safe categorical order (validated palette); text stays in ink colours.
COLOURS = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4',
               green='#008300', violet='#6250d6', red='#e34948', ink='#1f1f1f', muted='#52514e',
               grey='#8a8985', light='#c9c8c3')
C = COLOURS
PANEL_LABEL_SIZE = 7


def save(fig, name, out, formats, dpi):
    os.makedirs(out, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=dpi)
    plt.close(fig)
    print(f'  {name:15s} -> ' + ', '.join(os.path.relpath(os.path.join(out, f'{name}.{e}')) for e in formats))


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def parabolic_min(fr, s):
    """Sub-grid minimum of a sampled curve (three-point parabola); used for the 31-point baselines."""
    j = int(np.argmin(s))
    if 0 < j < len(s) - 1:
        y0, y1, y2 = s[j - 1], s[j], s[j + 1]
        den = y0 - 2 * y1 + y2
        if den > 1e-9:
            return fr[j] + 0.5 * (y0 - y2) / den * (fr[1] - fr[0])
    return fr[j]


def read_csv(name):
    with open(os.path.join(DATA, name)) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ============================== Fig. 2 ==============================
def fig_framework(out, formats, dpi):
    """ICP framework: data -> physically consistent representation -> PC-Net -> ICP inverse -> output."""
    W, H = COL2, 1.40
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

    def box(x, y, w, h, title, lines, edge, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.0,rounding_size=0.06', fc=fill, ec=edge, lw=1.0))
        ax.text(x + w / 2, y + h - 0.08, title, ha='center', va='top', fontsize=7.5, weight='bold', color=C['ink'])
        ax.text(x + 0.07, y + h - 0.28, '\n'.join(lines), ha='left', va='top', fontsize=6.6, color=C['ink'], linespacing=1.32)

    def arrow(x0, y0, x1, y1, col=C['muted']):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=0.9, color=col, shrinkA=0, shrinkB=0))

    y, h, gap = 0.03, 1.02, 0.13
    widths = [1.18, 1.58, 1.42, 1.62, 0.79]
    xs = [0.04]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    # --- box texts (edit freely) ---
    box(xs[0], y, widths[0], h, 'Full-wave data',
        ['500 HFSS designs', '9 free dimensions', '(Table I)', '31-point $S_{11}$, 4–7 GHz', '161 s per design'], C['muted'], '#f4f4f2')
    box(xs[1], y, widths[1], h, '1  Physically consistent data',
        ['Foster circuit fit per design,', 'all elements $\\geq 0$ (positive-real)', '$\\Rightarrow\\,|S_{11}|\\leq 1$ and causal',
         'order selected by frequency CV', '1 MHz super-resolved responses'], C['blue'], '#eef4fc')
    box(xs[2], y, widths[2], h, '2  PC-Net forward',
        ['dimensions $\\mathbf{x}\\rightarrow$ circuit', 'elements $\\rightarrow S_{11}(f)$', 'passive and causal for any $\\mathbf{x}$',
         '5-member ensemble: $\\mu,\\sigma$', 'calibrated bound $\\mu+\\kappa\\sigma$'], C['blue'], '#eef4fc')
    box(xs[3], y, widths[3], h, '3  ICP inverse',
        ['generator $g(f^{*},B,\\mathbf{z})$ trained', 'through the frozen PC-Net with', 'box, robust $-10$ dB band and',
         'data-support constraints;', 'test-time refinement'], C['orange'], '#fdf0ea')
    box(xs[4], y, widths[4], h, 'Output',
        ['top-$K$ diverse', 'designs on the', '0.01 mm grid', 'with predicted', '$S_{11}\\pm\\sigma$'], C['aqua'], '#e8f7f1')
    for i in range(4):
        arrow(xs[i] + widths[i], y + h / 2, xs[i + 1], y + h / 2)
    xm = xs[3] + widths[3] / 2
    ax.text(xm, y + h + 0.19, 'target $f^{*}$ (optional band $B$)', ha='center', va='bottom', fontsize=7, color=C['ink'])
    arrow(xm, y + h + 0.18, xm, y + h, col=C['orange'])
    save(fig, 'fig_framework', out, formats, dpi)


# ============================== Fig. 3 ==============================
def fig_extraction(out, formats, dpi):
    """Frequency hold-out test: Foster model fitted to the even samples predicts the odd (held-out) samples."""
    d = np.load(os.path.join(DATA, 'fig_extraction.npz'))
    f, S, ev, od = d['f'], d['S11'], d['even_idx'], d['odd_idx']
    fd, gp = d['f_dense'], d['S11_foster_dense']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL1, 1.75), gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.32))
    # (a) Smith-plane view
    th = np.linspace(0, 2 * np.pi, 300)
    a1.plot(np.cos(th), np.sin(th), color=C['grey'], lw=0.6)
    a1.plot(0.316 * np.cos(th), 0.316 * np.sin(th), color=C['grey'], lw=0.5, ls=':')   # |S11| = -10 dB circle
    a1.axhline(0, color=C['light'], lw=0.4); a1.axvline(0, color=C['light'], lw=0.4)
    a1.plot(gp.real, gp.imag, color=C['blue'], lw=1.0)
    a1.plot(S[ev].real, S[ev].imag, 'o', ms=2.6, mfc=C['ink'], mec=C['ink'], mew=0.4)
    a1.plot(S[od].real, S[od].imag, 'o', ms=3.2, mfc='white', mec=C['orange'], mew=0.8)
    a1.set_aspect('equal'); a1.set_xlim(-1.05, 1.05); a1.set_ylim(-1.05, 1.05)
    a1.set_xticks([-1, 0, 1]); a1.set_yticks([-1, 0, 1]); a1.grid(False)
    a1.set_xlabel('Re $S_{11}$'); a1.set_ylabel('Im $S_{11}$', labelpad=0)
    a1.text(0.03, 0.03, '(a)', transform=a1.transAxes, va='bottom', fontsize=PANEL_LABEL_SIZE)
    # (b) magnitude
    D = db(S)
    a2.plot(fd, d['S11dB_linear_interp'], color=C['grey'], lw=0.9, ls=(0, (4, 2)), label='linear interp.')
    a2.plot(fd, db(gp), color=C['blue'], lw=1.1, label='Foster (passive)')
    a2.plot(f[ev], D[ev], 'o', ms=2.6, mfc=C['ink'], mec=C['ink'], label='Full-wave: fit')
    a2.plot(f[od], D[od], 'o', ms=3.2, mfc='white', mec=C['orange'], mew=0.8, label='Full-wave: held out')
    a2.axhline(-10, color=C['muted'], lw=0.6, ls='--')
    a2.set_xlim(4, 7); a2.set_xlabel('Frequency (GHz)'); a2.set_ylabel('$|S_{11}|$ (dB)', labelpad=1)
    lo = min(D.min(), db(gp).min()); a2.set_ylim(np.floor(lo / 5) * 5 - 1, 1)
    a2.legend(loc='lower left', fontsize=5.6, handlelength=1.5, borderaxespad=0.3, labelspacing=0.3)
    a2.text(0.97, 0.03, '(b)', transform=a2.transAxes, va='bottom', ha='right', fontsize=PANEL_LABEL_SIZE)
    save(fig, 'fig_extraction', out, formats, dpi)


# ============================== Fig. 4 ==============================
def fig_forward(out, formats, dpi):
    """PC-Net on unseen geometries (out-of-fold predictions of the five-fold, geometry-level CV)."""
    d = np.load(os.path.join(DATA, 'fig_forward.npz'))
    f = d['f']
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 1.72), gridspec_kw=dict(width_ratios=[1.55, 0.95, 1.25], wspace=0.36))
    cols = [C['blue'], C['orange'], C['aqua']]
    # (a) three median-error designs: full-wave samples, PC-Net mean +- sigma, Extra Trees
    for j in range(len(d['picks'])):
        mu, sd = d['pcnet_mu'][j], d['pcnet_sd'][j]
        a1.fill_between(d['f_dense'], mu - sd, mu + sd, color=cols[j], alpha=0.18, lw=0)
        a1.plot(d['f_dense'], mu, color=cols[j], lw=1.1)
        a1.plot(f, d['extratrees_dB'][j], color=cols[j], lw=0.7, ls=(0, (2, 1.5)))
        a1.plot(f, d['hfss_dB'][j], 'o', ms=2.3, mfc='white', mec=cols[j], mew=0.7)
    a1.axhline(-10, color=C['muted'], lw=0.6, ls='--')
    a1.plot([], [], 'o', ms=2.3, mfc='white', mec='k', mew=0.7, label='Full-wave (unseen)')
    a1.plot([], [], '-', color='k', lw=1.1, label='PC-Net $\\mu\\pm\\sigma$')
    a1.plot([], [], color='k', lw=0.7, ls=(0, (2, 1.5)), label='Extra Trees')
    a1.legend(loc='lower left', fontsize=5.8, borderaxespad=0.2, labelspacing=0.3)
    a1.set_xlim(4, 7); a1.set_xlabel('Frequency (GHz)'); a1.set_ylabel('$|S_{11}|$ (dB)', labelpad=1)
    a1.text(0.98, 0.04, '(a)', transform=a1.transAxes, ha='right', fontsize=PANEL_LABEL_SIZE)
    # (b) resonance-frequency parity on matched designs
    m = d['matched'].astype(bool)
    t = d['fres_true'][m]
    gf = np.array([parabolic_min(f, s) for s in d['pred_dB_kriging']])[m]
    a2.plot([4, 7], [4, 7], color=C['grey'], lw=0.6)
    a2.scatter(t, gf, s=5, marker='x', color=C['grey'], lw=0.6, label='Kriging', zorder=2)
    a2.scatter(t, d['fres_pcnet'][m], s=6, color=C['blue'], edgecolor='white', linewidth=0.3, label='PC-Net', zorder=3)
    a2.set_xlim(4, 7); a2.set_ylim(4, 7); a2.set_aspect('equal')
    a2.set_xlabel('Full-wave $f_r$ (GHz)'); a2.set_ylabel('Predicted $f_r$ (GHz)', labelpad=1)
    a2.legend(loc='upper left', fontsize=5.8, borderaxespad=0.2, handletextpad=0.2)
    a2.text(0.97, 0.04, '(b)', transform=a2.transAxes, ha='right', fontsize=PANEL_LABEL_SIZE)
    # (c) screening precision@5 per queried frequency (mean over the five test folds)
    T, fold = d['S11dB_true'], d['fold']

    def p_at_5(P):
        res = np.full((5, len(f)), np.nan)
        for k in range(5):
            te = fold == k
            for j in range(len(f)):
                truth = T[te, j] <= -10
                if truth.any():
                    res[k, j] = truth[np.argsort(P[te, j])[:5]].mean()
        return 100 * np.nanmean(res, 0)
    base = np.array([100 * np.mean([np.mean(T[fold == k, j] <= -10) for k in range(5)]) for j in range(len(f))])
    for key, lab, col, ls, mk in [('pred_dB_pcnet', 'PC-Net', C['blue'], '-', 'o'),
                                  ('pred_dB_kriging', 'Kriging', C['grey'], '--', 's'),
                                  ('pred_dB_extratrees', 'Extra Trees', C['light'], ':', '^')]:
        a3.plot(f, p_at_5(d[key]), color=col, ls=ls, lw=1.0, marker=mk, ms=2.2, label=lab)
    a3.fill_between(f, 0, base, color=C['yellow'], alpha=0.35, lw=0, label='base rate')
    a3.set_xlim(4, 7); a3.set_ylim(0, 102); a3.set_xlabel('Queried frequency (GHz)'); a3.set_ylabel('Precision@5 (%)', labelpad=1)
    a3.legend(loc='upper right', fontsize=5.6, borderaxespad=0.2, labelspacing=0.25, ncol=2, columnspacing=0.8)
    a3.text(0.03, 0.04, '(c)', transform=a3.transAxes, fontsize=PANEL_LABEL_SIZE)
    save(fig, 'fig_forward', out, formats, dpi)


# ============================== Fig. 5 ==============================
def fig_inverse(out, formats, dpi):
    """Inverse design: amortisation, physical scaling of the designs, predicted responses of the final designs."""
    meta = json.load(open(os.path.join(DATA, 'fig_inverse_meta.json')))
    kap = meta['kappa']
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(COL2, 1.72), gridspec_kw=dict(width_ratios=[0.95, 1.15, 1.45], wspace=0.36))
    # (a) calibrated success of the top-ranked design versus refinement steps (mean +- s.e. over folds)
    am = read_csv('fig_inverse_amortisation.csv')
    ok = (am['worst_inband_dB'].astype(float) + kap * am['oracle_sd_at_f'].astype(float)) <= -10
    steps, fold = am['steps'].astype(int), am['fold'].astype(int)
    for init, col, mk, lab in [('generator', C['orange'], 'o', 'ICP generator init.'), ('random', C['grey'], 's', 'random init.')]:
        xs, ms, ses = [], [], []
        for s in sorted(set(steps)):
            per_fold = [ok[(am['init'] == init) & (steps == s) & (fold == k)].mean() for k in sorted(set(fold))]
            xs.append(s); ms.append(100 * np.mean(per_fold)); ses.append(100 * np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
        a1.errorbar(xs, ms, yerr=ses, color=col, marker=mk, ms=3.0, lw=1.1, capsize=1.5, elinewidth=0.6, label=lab)
    a1.set_xscale('symlog', linthresh=10); a1.set_xticks([0, 10, 30, 100]); a1.set_xticklabels(['0', '10', '30', '100'])
    a1.set_xlabel('Refinement steps'); a1.set_ylabel('Calibrated success (%)', labelpad=1); a1.set_ylim(0, 105)
    a1.legend(loc='lower right', fontsize=5.8, borderaxespad=0.2)
    a1.text(0.04, 0.04, '(a)', transform=a1.transAxes, va='bottom', fontsize=PANEL_LABEL_SIZE)
    # (b) patch length of the top-ranked design per target, against matched full-wave designs and f*L = const
    hf = read_csv('fig_inverse_hfss_matched.csv')
    gr = read_csv('fig_inverse_icp_grid.csv')
    a2.scatter(hf['f_res_GHz'].astype(float), hf['L_mm'].astype(float), s=4, color=C['light'], edgecolor='none',
               label='Full-wave designs ($\\leq-10$ dB)', zorder=1)
    ff = np.linspace(4, 7, 100)
    a2.plot(ff, meta['fL_hfss_mean'] / ff, color=C['muted'], lw=0.8, ls='--',
            label=f"$f\\,L$ = {meta['fL_hfss_mean']:.1f} GHz$\\cdot$mm", zorder=2)
    tgt, L, Cm = gr['target_GHz'].astype(float), gr['L_mm'].astype(float), gr['C_mm'].astype(float)
    thr = meta['family_threshold_C_mm']
    a2.scatter(tgt[Cm < thr], L[Cm < thr], s=10, color=C['blue'], edgecolor='white', linewidth=0.3,
               label=f'ICP, $C<{thr:g}$ mm', zorder=3)
    a2.scatter(tgt[Cm >= thr], L[Cm >= thr], s=11, marker='D', color=C['orange'], edgecolor='white', linewidth=0.3,
               label=f'ICP, $C\\geq{thr:g}$ mm', zorder=3)
    a2.axhline(meta['L_lower_bound_mm'], color=C['red'], lw=0.6, ls=':')
    a2.text(4.05, meta['L_lower_bound_mm'] + 0.12, 'lower bound of $L$', ha='left', va='bottom', fontsize=5.6, color=C['red'])
    a2.set_xlim(4, 7); a2.set_ylim(14.5, 23.2); a2.set_xlabel('Frequency (GHz)'); a2.set_ylabel('$L$ (mm)', labelpad=1)
    a2.legend(loc='upper right', fontsize=5.3, borderaxespad=0.2, labelspacing=0.25, handletextpad=0.3,
              frameon=True, facecolor='white', edgecolor='none', framealpha=0.92)
    a2.text(0.97, 0.09, '(b)', transform=a2.transAxes, va='bottom', ha='right', fontsize=PANEL_LABEL_SIZE)
    # (c) predicted |S11| (mean +- sigma) of the six frequency-targeted designs
    dz = np.load(os.path.join(DATA, 'fig_inverse_designs.npz'))
    cols = [C['blue'], C['orange'], C['aqua'], C['yellow'], C['magenta'], C['violet']]
    for j, t in enumerate(dz['targets']):
        a3.fill_between(dz['f'], dz['mu'][j] - dz['sd'][j], dz['mu'][j] + dz['sd'][j], color=cols[j], alpha=0.15, lw=0)
        a3.plot(dz['f'], dz['mu'][j], color=cols[j], lw=1.0, label=f'{t:.1f}')
        a3.axvline(t, color=cols[j], lw=0.5, ls=':')
    a3.axhline(-10, color=C['muted'], lw=0.6, ls='--')
    a3.set_xlim(4, 7); a3.set_xlabel('Frequency (GHz)'); a3.set_ylabel('Predicted $|S_{11}|$ (dB)', labelpad=1)
    a3.legend(loc='lower left', fontsize=5.6, ncol=3, title='$f^*$ (GHz)', title_fontsize=5.8, borderaxespad=0.2,
              columnspacing=0.8, handlelength=1.2)
    a3.text(0.97, 0.04, '(c)', transform=a3.transAxes, va='bottom', ha='right', fontsize=PANEL_LABEL_SIZE)
    save(fig, 'fig_inverse', out, formats, dpi)


FIGURES = {'framework': fig_framework, 'extraction': fig_extraction, 'forward': fig_forward, 'inverse': fig_inverse}

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Regenerate the paper figures from a figure_data folder')
    ap.add_argument('--only', nargs='*', choices=list(FIGURES), help='subset of figures (default: all)')
    ap.add_argument('--data', default=DATA, help='figure_data folder (default: runs/paper/figure_data)')
    ap.add_argument('--out', default=os.path.join(HERE, 'output'), help='output folder (default: figures/output)')
    ap.add_argument('--formats', nargs='*', default=['pdf', 'png'], help='file formats, e.g. pdf png svg')
    ap.add_argument('--dpi', type=int, default=300, help='raster resolution for PNG')
    a = ap.parse_args()
    DATA = os.path.abspath(a.data)
    if not os.path.isdir(DATA):
        raise SystemExit(f'figure data not found: {DATA}')
    for name in (a.only or list(FIGURES)):
        FIGURES[name](a.out, a.formats, a.dpi)
