#!/usr/bin/env python3
"""
Forward model: antenna dimensions -> predicted S11 (with calibrated uncertainty) and, optionally, the RF circuit.

Use it to check any geometry, for example a design of your own or one returned by design_antenna.py.

Dimensions are given in mm, in the HFSS order of the model:
    Air_gap  W_Patch  L_Patch  Sx  Sy  W_Patch1  WS  Curve_Amp1  Extra
    (paper symbols: G  W  L  Sx  Sy  W1  W_T  C  E)

Examples
  python predict_s11.py --dims 1.78 11.93 18.61 0.34 4.70 15.35 1.36 2.83 5.07
  python predict_s11.py --dims 1.78 11.93 18.61 0.34 4.70 15.35 1.36 2.83 5.07 --circuit
  python predict_s11.py --csv examples/5-6GHz_RL10/hfss_samples.csv      # every row of a samples-style CSV
  python predict_s11.py --csv ../data/hfss/samples.csv --rows 101 291     # simulated designs, HFSS curve overlaid

Outputs (in --out, default ./predictions):
  predictions.csv        resonance, minimum |S11|, mean and calibrated -10 dB (or --rl) bands, support distance
  predicted_<id>.s1p     predicted S11 (complex ensemble mean), Touchstone, 1 MHz grid
  predicted_S11.png      mean +- 1 sigma, with the HFSS response when the design is one of the 500 simulated ones
  rf_circuit_<id>.txt    equivalent circuit (with --circuit, needs scipy)
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from icp_engine import ICPModel, _band  # noqa: E402

HFSS_DIR = os.path.join(HERE, '..', 'data', 'hfss')


def read_hfss(run_id):
    p = os.path.join(HFSS_DIR, f'{run_id}.s1p')
    if not os.path.exists(p):
        return None
    d = np.array([[float(v) for v in ln.split()] for ln in open(p) if ln.strip() and ln[0] not in '!#'])
    return d[:, 0], 20 * np.log10(d[:, 1])


def load_designs(a, names):
    if a.dims:
        if len(a.dims) != 9:
            raise SystemExit('--dims needs 9 values: ' + ' '.join(names))
        return ['design_1'], np.array([a.dims], float)
    rows = list(csv.DictReader(open(a.csv)))
    if a.rows:
        rows = [rows[i] for i in a.rows]
    ids = [r.get('run_id') or r.get('design_id') or f'design_{i + 1}' for i, r in enumerate(rows)]
    X = np.array([[float(r[n]) for n in names] for r in rows])
    return ids, X


def main():
    ap = argparse.ArgumentParser(description='Antenna dimensions -> predicted S11 (PC-Net ensemble, NumPy)')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--dims', type=float, nargs='+', metavar='MM',
                     help='Air_gap W_Patch L_Patch Sx Sy W_Patch1 WS Curve_Amp1 Extra (mm)')
    src.add_argument('--csv', help='CSV with one design per row and the 9 HFSS column names (samples.csv format)')
    ap.add_argument('--rows', type=int, nargs='*', help='row numbers of --csv to use (0-based; default: all)')
    ap.add_argument('--rl', type=float, default=10.0, help='return loss that defines the band, in dB (default 10)')
    ap.add_argument('--circuit', action='store_true', help='also fit the RF equivalent circuit (needs scipy)')
    ap.add_argument('--out', default='predictions', help='output folder')
    ap.add_argument('--model', default=None, help='model folder (default: designer/model)')
    a = ap.parse_args()

    model = ICPModel(a.model)
    names = model.params
    ids, X = load_designs(a, names)
    outside = (X < model.LB - 1e-9) | (X > model.UB + 1e-9)
    U = model.to_unit(X)
    thr = -abs(a.rl)
    fg = np.round(np.arange(4.0, 7.0 + 1e-9, 0.001), 3)
    mu, sd = model.s11_db_stats(U, fg)
    rob = mu + model.kappa * sd
    dist = model.support_dist(U)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    print(f'\nForward prediction of {len(ids)} design(s); band defined by |S11| <= {thr:.0f} dB; calibrated bound = mu + {model.kappa:.2f} sigma')
    for i, rid in enumerate(ids):
        j = int(np.argmin(mu[i]))
        lo, hi = _band(fg, mu[i], j, thr)
        rlo, rhi = _band(fg, rob[i], int(np.argmin(rob[i])), thr)
        notes = []
        if outside[i].any():
            notes.append('OUTSIDE the sampled ranges: ' + ', '.join(n for n, o in zip(names, outside[i]) if o))
        if dist[i] > model.dmax:
            notes.append('outside the data support (extrapolation)')
        r = dict(design_id=rid, **{n: round(float(v), 3) for n, v in zip(names, X[i])},
                 pred_resonance_GHz=round(float(fg[j]), 3), pred_S11_min_dB=round(float(mu[i, j]), 2),
                 pred_S11_std_at_resonance_dB=round(float(sd[i, j]), 2),
                 band_lo_GHz=None if np.isnan(lo) else round(float(lo), 3), band_hi_GHz=None if np.isnan(hi) else round(float(hi), 3),
                 band_MHz=0.0 if np.isnan(lo) else round(1000 * float(hi - lo)),
                 calibrated_band_lo_GHz=None if np.isnan(rlo) else round(float(rlo), 3),
                 calibrated_band_hi_GHz=None if np.isnan(rhi) else round(float(rhi), 3),
                 support_distance=round(float(dist[i]), 3), notes='; '.join(notes) or 'inside the data support')
        hf = read_hfss(rid)
        if hf is not None:
            fh, sh = hf
            pm = np.interp(fh, fg, mu[i])
            r['hfss_available'] = True
            r['MAE_vs_HFSS_dB'] = round(float(np.abs(pm - sh).mean()), 2)
            r['notes'] += '; one of the 500 training designs (in-sample)'
        rows.append(r)
        band = f"{r['band_lo_GHz']}-{r['band_hi_GHz']} GHz" if r['band_lo_GHz'] is not None else 'none'
        print(f"  {rid}: resonance {r['pred_resonance_GHz']:.3f} GHz, |S11|min {r['pred_S11_min_dB']:.1f} +- {r['pred_S11_std_at_resonance_dB']:.1f} dB,"
              f" {thr:.0f} dB band {band} | {r['notes']}" + (f" | MAE vs HFSS {r['MAE_vs_HFSS_dB']} dB" if hf is not None else ''))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(a.out, 'predictions.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    S = model.s11(U, fg).mean(0)
    for i, rid in enumerate(ids):
        with open(os.path.join(a.out, f'predicted_{rid}.s1p'), 'w') as fh:
            fh.write('! ICP forward prediction (PC-Net ensemble mean, passive and causal)\n')
            for n, v in zip(names, X[i]):
                fh.write(f'!   {n} = {v:g}mm\n')
            fh.write('# GHz S MA R 50\n')
            for fv, s in zip(fg, S[i]):
                fh.write(f'{fv:.3f} {abs(s):.8f} {np.degrees(np.angle(s)):.5f}\n')
    if a.circuit:
        from rf_circuit import describe, fit_equivalent_circuit
        f10 = np.round(np.arange(4.0, 7.0 + 1e-9, 0.01), 2)
        S10 = model.s11(U, f10).mean(0)
        for i, rid in enumerate(ids):
            txt = describe(fit_equivalent_circuit(f10, S10[i]))
            print(f'\nRF equivalent circuit of {rid}:\n{txt}')
            with open(os.path.join(a.out, f'rf_circuit_{rid}.txt'), 'w') as fh:
                fh.write(f'RF equivalent circuit of {rid} (fitted to the predicted S11, 4-7 GHz, Z0 = 50 ohm)\n{txt}\n')
    _plot(fg, mu, sd, ids, thr, os.path.join(a.out, 'predicted_S11.png'))
    print(f'\nFiles written to {os.path.abspath(a.out)}')


def _plot(fg, mu, sd, ids, thr, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    for i, rid in enumerate(ids[:6]):
        c = cols[i % len(cols)]
        ax.fill_between(fg, mu[i] - sd[i], mu[i] + sd[i], color=c, alpha=0.15, lw=0)
        ax.plot(fg, mu[i], color=c, lw=1.6, label=f'{rid} (predicted)')
        hf = read_hfss(rid)
        if hf is not None:
            ax.plot(hf[0], hf[1], 'o', color=c, ms=3, label=f'{rid} (HFSS)')
    ax.axhline(thr, color='0.35', lw=0.9, ls='--')
    ax.set_xlim(4, 7)
    ax.set_ylim(min(-35, np.nanmin(mu[:6]) - 2), 0.5)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('|S$_{11}$| (dB)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False, loc='lower left')
    ax.set_title('ICP forward prediction (mean $\\pm$1$\\sigma$)', fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
