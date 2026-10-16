#!/usr/bin/env python3
"""
ICP inverse antenna designer  -  give an S-parameter target, get HFSS-ready antenna dimensions.

A target is a frequency (or a band) and a return loss:
  --freq F                 centre frequency in GHz (4-7)
  --band F1 F2             required band in GHz; equivalent to --freq (F1+F2)/2 --bw F2-F1
  --bw B                   required bandwidth in GHz around --freq (default: widest band the model can centre on F)
  --rl RL                  required return loss in dB over the band (default 10, i.e. |S11| <= -10 dB; 15 -> |S11| <= -15 dB)
  --circuit                also fit the RF equivalent circuit (R, L, C, Q, resonances) of each design  [needs scipy]

Examples
  python design_antenna.py                                  # interactive prompt
  python design_antenna.py --freq 5.5                       # widest calibrated -10 dB band centred on 5.5 GHz
  python design_antenna.py --band 5 6 --rl 10 --circuit     # |S11| <= -10 dB from 5 to 6 GHz
  python design_antenna.py --band 5 6 --rl 15 --circuit     # |S11| <= -15 dB from 5 to 6 GHz
  python design_antenna.py --freq 4.8 5.2 6.0 --top 3 --out my_designs

Outputs (in --out):
  designs.csv            9 free + 2 derived + 2 fixed variables, predicted performance against the target
  hfss_samples.csv       same column layout as the original samples.csv (for the HFSS sweep script)
  predicted_<id>.s1p     predicted S11, Touchstone, 1 MHz grid
  predicted_S11.png      predicted S11 with ensemble uncertainty, target band and threshold
  rf_circuit_<id>.txt    equivalent circuit of each design (with --circuit)
Requirements: numpy (matplotlib optional for the plot, scipy only for --circuit). No PyTorch needed.
"""
import argparse, os, sys, time, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icp_engine import ICPModel

FIXED = {'H_sub': 0.79, 'Curve_len1': 1.0}
PAPER = {'Air_gap': 'G', 'W_Patch': 'W', 'L_Patch': 'L', 'Sx': 'Sx', 'Sy': 'Sy', 'W_Patch1': 'W1', 'WS': 'W_T',
         'Curve_Amp1': 'C', 'Extra': 'E', 'L_Patch1': 'L1', 'Curve_pos': 'pos'}


def confidence(ev, i, thr, envelope=None):
    if not ev['in_support'][i]:
        return 'LOW (outside simulated design space)'
    if ev['worst_rob_inband'][i] <= thr:
        c = 'HIGH' if ev['S11_sd_at_f'][i] < 2.0 else 'MEDIUM'
    elif ev['worst_mu_inband'][i] <= thr:
        c = 'MEDIUM-LOW (mean prediction meets the target, calibrated bound does not)'
    else:
        return 'LOW (target not met by the prediction; closest design found)'
    widest = _widest(envelope, thr)
    if widest is not None and ev['mean_BW'][i] > widest + 1e-9:
        c += f' | EXTRAPOLATION: predicted band is wider than any of the 500 simulated designs ({widest:.2f} GHz at {thr:.0f} dB); check with one full-wave run'
    return c


def _widest(envelope, thr):
    """Widest contiguous band below thr among the 500 simulated designs (from model/config.json)."""
    if not envelope:
        return None
    keys = sorted(envelope, key=lambda k: abs(float(k) - thr))
    k = keys[0]
    return envelope[k] if abs(float(k) - thr) < 1e-6 else None


def run(model, fstar, bw, top, out, seed, rl=10.0, circuit=False, label=None):
    thr = -abs(rl)
    t0 = time.time()
    res = model.design(fstar, bw=bw, K=top, seed=seed, thr=thr)
    dt = time.time() - t0
    os.makedirs(out, exist_ok=True)
    U, ev = res['U'], res['ev']
    X = model.from_unit(U)
    cpos, lp1 = model.derived(X)
    names = model.params
    band_lo, band_hi = fstar - res['bw_req'] / 2, fstar + res['bw_req'] / 2
    tag = label or f'{fstar:.2f}GHz'
    rows = []
    for i in range(len(U)):
        r = dict(design_id=f'ICP_{tag}_{i + 1}', target_GHz=round(fstar, 4),
                 requested_band_GHz='max centred' if res['maximise_bw'] else f'{band_lo:.3f}-{band_hi:.3f}',
                 required_S11_dB=thr, required_return_loss_dB=abs(rl))
        r.update({n: round(float(v), 2) for n, v in zip(names, X[i])})
        r.update(Curve_pos=round(float(cpos[i]), 3), L_Patch1=round(float(lp1[i]), 2), **FIXED)
        r.update(pred_S11_at_centre_dB=round(float(ev['S11_at_f'][i]), 2), pred_S11_std_dB=round(float(ev['S11_sd_at_f'][i]), 2),
                 worst_S11_in_band_mean_dB=round(float(ev['worst_mu_inband'][i]), 2),
                 worst_S11_in_band_calibrated_dB=round(float(ev['worst_rob_inband'][i]), 2),
                 band_coverage_mean_pct=round(100 * float(ev['coverage_mean'][i]), 1),
                 band_coverage_calibrated_pct=round(100 * float(ev['coverage_robust'][i]), 1),
                 pred_resonance_GHz=round(float(ev['f_res'][i]), 3),
                 pred_band_lo_GHz=_r(ev['mean_lo'][i]), pred_band_hi_GHz=_r(ev['mean_hi'][i]),
                 pred_BW_MHz=round(1000 * float(ev['mean_BW'][i]), 0),
                 calibrated_band_lo_GHz=_r(ev['robust_lo'][i]), calibrated_band_hi_GHz=_r(ev['robust_hi'][i]),
                 calibrated_BW_MHz=round(1000 * float(ev['robust_BW'][i]), 0),
                 support_distance=round(float(ev['support_dist'][i]), 3),
                 confidence=confidence(ev, i, thr, model.cfg.get('widest_band_in_data_GHz')))
        rows.append(r)
    _write_csv(os.path.join(out, 'designs.csv'), rows)
    _write_csv(os.path.join(out, 'hfss_samples.csv'),
               [dict(run_id=r['design_id'], **{n: r[n] for n in ['Air_gap', 'W_Patch', 'L_Patch', 'Sx', 'Sy', 'W_Patch1', 'WS', 'Curve_Amp1', 'Extra']})
                for r in rows])
    fg = np.round(np.arange(4.0, 7.0 + 1e-9, 0.001), 3)
    S = model.s11(U, fg).mean(0)  # ensemble-mean complex response (passive and causal)
    for i, r in enumerate(rows):
        with open(os.path.join(out, f"predicted_{r['design_id']}.s1p"), 'w') as fh:
            fh.write(f"! ICP-predicted S11, target {r['requested_band_GHz']} GHz at {thr:.0f} dB. Complex ensemble mean (passive and causal);\n")
            fh.write("! its |S11| can differ slightly from the dB-domain mean reported in designs.csv.\n")
            for n in names:
                fh.write(f"!   {n} = {r[n]}mm\n")
            fh.write('# GHz S MA R 50\n')
            for fv, s in zip(fg, S[i]):
                fh.write(f'{fv:.3f} {abs(s):.8f} {np.degrees(np.angle(s)):.5f}\n')
    circuits = []
    if circuit:
        from rf_circuit import fit_equivalent_circuit, describe
        f10 = np.round(np.arange(4.0, 7.0 + 1e-9, 0.01), 2)
        S10 = model.s11(U, f10).mean(0)
        for i, r in enumerate(rows):
            c = fit_equivalent_circuit(f10, S10[i])
            txt = describe(c)
            circuits.append(txt)
            with open(os.path.join(out, f"rf_circuit_{r['design_id']}.txt"), 'w') as fh:
                fh.write(f"RF equivalent circuit of {r['design_id']} (fitted to the predicted S11, 4-7 GHz, Z0 = 50 ohm)\n{txt}\n")
    _plot(ev, rows, fstar, res['bw_req'], thr, os.path.join(out, 'predicted_S11.png'))
    # ---------------- console report ----------------
    req = 'widest band centred on target' if res['maximise_bw'] else f'{band_lo:.3f}-{band_hi:.3f} GHz'
    print(f'\n=== Target: {req} with |S11| <= {thr:.0f} dB (return loss >= {abs(rl):.0f} dB) | {len(rows)} designs | {dt:.1f} s ===')
    print('Dimensions in mm (paper symbol / HFSS variable):')
    hdr = names + ['Curve_pos', 'L_Patch1']
    print('  '.join(f'{PAPER[h]:>8s}' for h in hdr))
    for r in rows:
        print('  '.join(f'{r[h]:>8.2f}' for h in hdr), f"   {r['design_id']}")
    print('  (HFSS names: ' + ', '.join(f'{PAPER[h]}={h}' for h in hdr) + ')\n')
    for i, r in enumerate(rows):
        print(f"{r['design_id']}: worst |S11| in band = {r['worst_S11_in_band_mean_dB']:.1f} dB (calibrated bound "
              f"{r['worst_S11_in_band_calibrated_dB']:.1f} dB) | band meeting target: {r['band_coverage_mean_pct']:.0f}% "
              f"(calibrated {r['band_coverage_calibrated_pct']:.0f}%) | {thr:.0f} dB band around centre "
              + (f"{r['pred_band_lo_GHz']}-{r['pred_band_hi_GHz']} GHz" if r['pred_band_lo_GHz'] is not None
                 else 'none (not matched at the centre frequency)') + f" | confidence {r['confidence']}")
        if circuits:
            print(circuits[i])
    print(f'Files written to {os.path.abspath(out)}')
    print('Fixed: H_sub = 0.79 mm, Curve_len1 = 1.  Derived: Curve_pos = 0.5*L_Patch + 2, L_Patch1 = W_Patch1 + Extra.')
    print('All values are surrogate predictions with calibrated uncertainty.')
    return rows


def _r(v):
    return None if not np.isfinite(v) else round(float(v), 3)


def _write_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot(ev, rows, fstar, bw, thr, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    ls = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    fg = ev['fgrid']
    if bw > 0:
        ax.axvspan(fstar - bw / 2, fstar + bw / 2, color='0.85', alpha=0.5, lw=0, label='required band')
    for i, r in enumerate(rows):
        ax.fill_between(fg, ev['mu'][i] - ev['sd'][i], ev['mu'][i] + ev['sd'][i], color=cols[i % 5], alpha=0.15, lw=0)
        ax.plot(fg, ev['mu'][i], color=cols[i % 5], ls=ls[i % 5], lw=1.6, label=r['design_id'])
    ax.axhline(thr, color='0.35', lw=0.9, ls='--')
    ax.axvline(fstar, color='0.35', lw=0.9, ls=':')
    ax.set_xlim(4, 7); ax.set_ylim(min(-35, np.nanmin(ev['mu']) - 2), 0.5)
    ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('|S$_{11}$| (dB)')
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=7, frameon=False, loc='lower left')
    span = f'{fstar - bw / 2:.2f}-{fstar + bw / 2:.2f} GHz' if bw > 0 else f'f* = {fstar:.2f} GHz'
    ax.set_title(f'ICP designs for {span}, |S$_{{11}}$| $\\leq$ {thr:.0f} dB (mean $\\pm$1$\\sigma$)', fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='S-parameter target -> antenna dimensions (ICP inverse model)')
    ap.add_argument('--freq', type=float, nargs='*', help='target centre frequency/frequencies in GHz (4-7)')
    ap.add_argument('--band', type=float, nargs=2, metavar=('F1', 'F2'), help='required band in GHz, e.g. --band 5 6')
    ap.add_argument('--bw', type=float, default=None, help='required bandwidth in GHz around --freq (default: maximise)')
    ap.add_argument('--rl', type=float, default=10.0, help='required return loss in dB over the band (10 -> |S11| <= -10 dB)')
    ap.add_argument('--circuit', action='store_true', help='also fit the RF equivalent circuit of each design (needs scipy)')
    ap.add_argument('--top', type=int, default=5, help='number of diverse designs to return')
    ap.add_argument('--out', type=str, default=None, help='output folder')
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--model', type=str, default=None, help='model folder (default ./model)')
    a = ap.parse_args()
    model = ICPModel(a.model)
    if a.band:
        f1, f2 = sorted(a.band)
        label = f'{f1:g}-{f2:g}GHz_RL{a.rl:g}'
        run(model, (f1 + f2) / 2, f2 - f1, a.top, a.out or f'designs_{label}', a.seed, a.rl, a.circuit, label)
        return
    freqs = a.freq
    if not freqs:
        s = input('Target: a centre frequency (e.g. 5.5) or a band (e.g. 5-6) in GHz: ').strip().replace(' ', '')
        r = input('Required return loss in dB [Enter = 10]: ').strip()
        a.rl = float(r) if r else 10.0
        if not a.circuit:
            a.circuit = input('Also fit the RF equivalent circuit (needs scipy)? [y/N]: ').strip().lower().startswith('y')
        if '-' in s:
            f1, f2 = sorted(float(v) for v in s.split('-'))
            label = f'{f1:g}-{f2:g}GHz_RL{a.rl:g}'
            run(model, (f1 + f2) / 2, f2 - f1, a.top, a.out or f'designs_{label}', a.seed, a.rl, a.circuit, label)
            return
        freqs = [float(v) for v in s.split(',')]
        b = input('Required bandwidth in GHz [Enter = maximise]: ').strip()
        a.bw = float(b) if b else None
    for fs in freqs:
        out = a.out or f'designs_{fs:.2f}GHz'
        if len(freqs) > 1 and a.out:
            out = os.path.join(a.out, f'{fs:.2f}GHz')
        run(model, fs, a.bw, a.top, out, a.seed, a.rl, a.circuit)


if __name__ == '__main__':
    main()
