"""Step 14 - paper tables and every number quoted in the text.

Output (in tables/):
    table2_forward_models.tex/.csv      Table II  exactly as in the paper
    table3_inverse_benchmark.tex/.csv   Table III exactly as in the paper
    table4_final_designs.tex/.csv       Table IV  exactly as in the paper
    table1_super_resolution.*, table2_forward_models_all_metrics.csv, table3_inverse_all_metrics.csv,
    inverse_top1_with_optimism.csv      extended versions with every metric
    paper_numbers.json                  all metrics in full precision (machine readable)
    paper_text_checks.json              every number quoted in the paper text, formatted as in the paper, together
                                        with the sentence fragment that contains it. tests/ and `run.py compare` use it.
"""
import glob
import json
import math
import os
import pickle
from decimal import ROUND_HALF_UP, Decimal

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

import circuit as C
from common import *
from inverse import Ensemble

OUT = 'tables'
os.makedirs(OUT, exist_ok=True)
N = {}       # full-precision numbers
CHECK = {}   # formatted numbers as they appear in the paper text


def tex(df, path, colfmt):
    with open(path, 'w') as fh:
        fh.write('\\begin{tabular}{' + colfmt + '}\n\\toprule\n')
        fh.write(' & '.join(df.columns) + ' \\\\\n\\midrule\n')
        for _, r in df.iterrows():
            fh.write(' & '.join(str(v) for v in r.values) + ' \\\\\n')
        fh.write('\\bottomrule\n\\end{tabular}\n')


def half_up(v, nd=2):
    """Round half up (4.745 -> 4.75), as in the paper tables."""
    return str(Decimal(str(round(float(v), 6))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def check(key, text, machine_dependent=False):
    CHECK[key] = dict(text=text, machine_dependent=machine_dependent)


WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

# ============================== data and super-resolution (Sections III-IV.A) ==============================
audit = json.load(open('audit.json')) if os.path.exists('audit.json') else {}
rt = audit.get('hfss_runtime')
if rt:
    N['hfss_runtime'] = rt
    check('hfss_runtime', f"taking {rt['mean_s']:.0f}~s each on average ({rt['total_h']:.1f}~h in total)")

S = pd.read_csv('interp_benchmark_summary.csv').set_index('method')
fq = json.load(open('fit_quality.json'))
order = ['Physics-consistent Foster model', 'Cubic spline (complex)', 'Linear (complex)', 'PCHIP (dB)', 'Cubic spline (dB)', 'Linear (dB)']
names1 = {'Physics-consistent Foster model': '\\textbf{Foster (passive)}', 'Cubic spline (complex)': 'Spline, complex',
          'Linear (complex)': 'Linear, complex', 'PCHIP (dB)': 'PCHIP, dB', 'Cubic spline (dB)': 'Spline, dB', 'Linear (dB)': 'Linear, dB'}
t1 = pd.DataFrame({'Interpolant': [names1[m] for m in order],
                   'MAE': [f"{S.loc[m, 'dB_MAE']:.2f}" for m in order],
                   'MAE res.': [f"{S.loc[m, 'dB_MAE_resonant']:.2f}" for m in order],
                   'P95 max': [f"{S.loc[m, 'dB_maxerr_p95']:.1f}" for m in order],
                   'Flips': [int(S.loc[m, 'threshold_flips']) for m in order],
                   '$|S_{11}|{>}1$': [int(S.loc[m, 'passivity_violations']) for m in order]})
t1.to_csv(f'{OUT}/table1_super_resolution.csv', index=False)
tex(t1, f'{OUT}/table1_super_resolution.tex', 'lccccc')
N['interp'] = {m: S.loc[m].to_dict() for m in order}
N['fit_quality'] = fq
fo, others = 'Physics-consistent Foster model', order[1:]
check('sr_mae', f"mean absolute error (MAE) of {S.loc[fo, 'dB_MAE']:.2f}~dB, against {S.loc[others, 'dB_MAE'].min():.2f}--"
                f"{S.loc[others, 'dB_MAE'].max():.2f}~dB for linear, spline and piecewise cubic interpolants")
check('sr_resonant', f"its error of {S.loc[fo, 'dB_MAE_resonant']:.2f}~dB is "
                     f"{S.loc[others, 'dB_MAE_resonant'].min() / S.loc[fo, 'dB_MAE_resonant']:.1f} times lower")
check('sr_flips', f"It makes {int(S.loc[fo, 'threshold_flips'])} wrong $-10$~dB decisions over the 500 designs, against at least "
                  f"{int(S.loc[others, 'threshold_flips'].min())} for the interpolants")
splines_violate = S.loc['Cubic spline (complex)', 'passivity_violations'] > 0 and S.loc['Cubic spline (dB)', 'passivity_violations'] > 0
check('sr_passivity', 'never violates passivity, whereas both cubic splines do' if S.loc[fo, 'passivity_violations'] == 0 and splines_violate
      else f"Foster violations {S.loc[fo, 'passivity_violations']}")

pr = np.load('processed.npz', allow_pickle=True)
f31, G31 = pr['f'], pr['G']
D31 = db(G31)
DS = pd.read_csv('design_descriptors.csv')
fits = pickle.load(open('circuit_fits.pkl', 'rb'))
i101 = int(np.where(pr['ids'].astype(str) == 'run_0101')[0][0])
K101 = int(pr['Ksel'][i101])
j59 = int(np.argmin(abs(f31 - 5.9)))
ev = np.arange(0, 31, 2)
s_hfss = D31[i101, j59]
s_lin = np.interp(f31[j59], f31[ev], D31[i101, ev])
s_fos = db(C.gamma_model(fits[i101][K101]['p_even'], np.array([f31[j59]]), K101))[0]
check('sr_example', f"the full-wave result is ${s_hfss:.1f}$~dB, linear interpolation gives ${s_lin:.1f}$~dB and the equivalent "
                    f"circuit predicts ${s_fos:.1f}$~dB")
n_dense, n_raw, n_deep = int(DS.matched.sum()), int((D31.min(1) <= -10).sum()), int(((DS.raw_min_dB - DS.S11_min_dB) > 3).sum())
N['matched'] = dict(dense=n_dense, raw=n_raw, deeper_than_best_sample_by_3dB=n_deep)
check('matched', f"{n_dense} designs meet $-10$~dB on the 1~MHz grid against {n_raw} on the raw grid, and {n_deep} designs have a "
                 f"true minimum more than 3~dB below their best sample")

# ============================== Table II: forward models ==============================
M = pd.read_csv('fwd_cv_mean.csv').set_index('model')
SD = pd.read_csv('fwd_cv_std.csv').set_index('model')
N['forward_cv_mean'] = M.to_dict(orient='index')
N['forward_cv_std'] = SD.to_dict(orient='index')
cal = json.load(open('calibration.json'))
N['calibration'] = cal
pc = 'PC-Net ensemble (proposed)'
paper_rows = [('ExtraTrees (dB)', 'Extra Trees'), ('Kriging/GP (dB)', 'Kriging'), ('MLP ensemble (dB)', 'MLP, dB'),
              ('MLP ensemble (Re/Im)', 'MLP, Re/Im'), (pc, 'PC-Net')]
rows = []
for m, nm in paper_rows:
    if m not in M.index:
        continue
    cells = [f"{M.loc[m, 'dB MAE']:.2f}$\\pm${SD.loc[m, 'dB MAE']:.2f}", f"{M.loc[m, 'f_res MAE (MHz)']:.0f}", f"{M.loc[m, 'F1']:.2f}",
             f"{M.loc[m, 'screen precision@5 (%)']:.0f}", f"{M.loc[m, 'passivity violations (%)']:.1f}"]
    if m == pc:
        nm, cells = f'\\textbf{{{nm}}}', [f'\\textbf{{{c}}}' for c in cells]
    rows.append([nm] + cells)
t2 = pd.DataFrame(rows, columns=['Model', 'MAE (dB)', '$f_r$ err. (MHz)', 'F1', 'P@5 (\\%)', '$|S_{11}|{>}1$ (\\%)'])
t2.to_csv(f'{OUT}/table2_forward_models.csv', index=False)
tex(t2, f'{OUT}/table2_forward_models.tex', 'lccccc')
M.T.to_csv(f'{OUT}/table2_forward_models_all_metrics.csv')

base = [m for m in ['ExtraTrees (dB)', 'Kriging/GP (dB)', 'MLP ensemble (dB)', 'MLP ensemble (Re/Im)'] if m in M.index]
B = M.loc[base]
dB_models = [m for m in base if m != 'MLP ensemble (Re/Im)']
check('fwd_mae', f"PC-Net lowers the MAE from {B['dB MAE'].min():.2f} to {M.loc[pc, 'dB MAE']:.2f}~dB, the error near resonance from "
                 f"{B['dB MAE (resonant pts)'].min():.2f} to {M.loc[pc, 'dB MAE (resonant pts)']:.2f}~dB and the resonance frequency error "
                 f"from {B['f_res MAE (MHz)'].min():.0f} to {M.loc[pc, 'f_res MAE (MHz)']:.0f}~MHz, with "
                 f"{M.loc[pc, 'f_res within 50 MHz (%)']:.0f}\\% of the resonances predicted within 50~MHz")
check('fwd_f1', f"(F1 score $\\le {M.loc[dB_models, 'F1'].max():.2f}$), whereas PC-Net reaches an F1 score of {M.loc[pc, 'F1']:.2f}")
check('fwd_screening', f"The picks of PC-Net meet $-10$~dB in {M.loc[pc, 'screen precision@5 (%)']:.0f}\\% of cases, against "
                       f"{B['screen precision@5 (%)'].max():.0f}\\% for the best baseline and a {M.loc[pc, 'screen base rate (%)']:.1f}\\% "
                       f"base rate, and the calibrated rule $\\mu+\\kappa\\sigma\\le-10$~dB is correct in "
                       f"{100 * cal['precision_robust_rule']:.1f}\\% of its decisions")
if 'MLP ensemble (Re/Im)' in M.index:
    check('fwd_passivity', f"violates passivity in {M.loc['MLP ensemble (Re/Im)', 'passivity violations (%)']:.1f}\\% of its predictions")
check('kappa', f"with $\\kappa={cal['kappa']:.2f}$ is an empirical one-sided 84\\% bound")
if 'Kriging/GP (dB)' in M.index:
    check('kriging_recall', f"since kriging detects fewer than {math.ceil(100 * M.loc['Kriging/GP (dB)', 'recall'])}\\% of the matched samples")

# ============================== Table III: inverse benchmark ==============================
R = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('inv_eval_rows_*.csv'))])
cv = R[R.method.str.startswith('curve:')].copy()
R = R[~R.method.str.startswith('curve:')].reset_index(drop=True)
top1 = R[R['rank'] == 0].copy()
# self-assessed prediction of the fold model that proposed the design -> optimism = reference - self (surrogate exploitation)
self_s = np.full(len(top1), np.nan)
for k in sorted(top1.fold.unique()):
    idx = np.where(top1.fold.values == k)[0]
    ens = Ensemble(torch.load(f'pcnet_fold{k}.pt'))
    Uq = to_unit(top1[PARAMS].values[idx])
    fq_ = top1.f_star.values[idx]
    for j, i in enumerate(idx):
        self_s[i] = db(ens.complex_np(Uq[j:j + 1], np.array([fq_[j]])))[:, 0, 0].mean()
top1['self_S11'] = self_s
top1['optimism_dB'] = top1.S11_at_f - top1.self_S11
kap = cal['kappa']
dmax_all = json.load(open('support.json'))['dmax_all']
meths = [('Catalogue retrieval (training HFSS designs)', 'Catalogue retrieval'), ('Direct inverse MLP', 'Direct inverse MLP'),
         ('Conditional VAE (data only)', 'Conditional VAE'), ('Tandem network (unconstrained)', 'Tandem (unconstrained)'),
         ('Constrained surrogate optimisation (random starts)', 'Constr. opt., random init.'),
         ('Ablation: ICP generator only (no refinement)', 'ICP generator only'),
         ('Ablation: no robust/support constraints', 'ICP w/o constraints'), ('Proposed ICP inverse', 'ICP (proposed)')]
inv, rows, rows_all = {}, [], []
for m, nm in meths:
    a, b = top1[top1.method == m], R[R.method == m]
    if len(a) == 0:
        continue
    hs, gr = a[a.kind == 'heldout_spec'], a[a.kind == 'grid']
    grp = b.groupby(['fold', 'target', 'kind', 'f_star'])
    div = grp[PARAMS].apply(lambda g: np.nan if len(g) < 2 else np.mean(
        [np.abs(to_unit(g.values[i]) - to_unit(g.values[j])).mean() for i in range(len(g)) for j in range(i + 1, len(g))])).mean()
    d = dict(spec_success=100 * hs.success.mean(), grid_success=100 * gr.success.mean(),
             calib_success=100 * ((a.worst_inband_dB + kap * a.oracle_sd_at_f) <= -10).mean(),
             spec_success_gp=100 * hs.success_gp.mean(), grid_success_gp=100 * gr.success_gp.mean(),
             top5_success=100 * grp.success.max().mean(), f_err_grid_median=gr.f_res_err_MHz.median(),
             f_err_all_median=a.f_res_err_MHz.median(), S11_oracle_median=a.S11_at_f.median(), optimism_median=a.optimism_dB.median(),
             in_support=100 * (a.support_dist <= dmax_all).mean(), oracle_sd=a.oracle_sd_at_f.mean(), time=a.time_s.mean(),
             diversity=div, n_designs=grp.size().mean())
    inv[m] = d
    opt = '--' if m.startswith('Catalogue') else f"{d['optimism_median']:.1f}".replace('-0.0', '0.0')
    t = '$<$0.01' if d['time'] < 0.01 else f"{d['time']:.2f}"
    cells = [f"{d['spec_success']:.0f}", f"{d['calib_success']:.0f}", opt, t]
    if m == 'Proposed ICP inverse':
        nm, cells = f'\\textbf{{{nm}}}', [f'\\textbf{{{cells[0]}}}', f'\\textbf{{{cells[1]}}}', cells[2], cells[3]]
    rows.append([nm] + cells)
    rows_all.append(dict(method=m, **d))
t3 = pd.DataFrame(rows, columns=['Method', 'Spec. (\\%)', 'Calib. (\\%)', 'Opt. (dB)', 'Time (s)'])
t3.to_csv(f'{OUT}/table3_inverse_benchmark.csv', index=False)
tex(t3, f'{OUT}/table3_inverse_benchmark.tex', 'lcccc')
pd.DataFrame(rows_all).to_csv(f'{OUT}/table3_inverse_all_metrics.csv', index=False)
top1.to_csv(f'{OUT}/inverse_top1_with_optimism.csv', index=False)
N['inverse'] = inv
icp = inv['Proposed ICP inverse']
N['inverse_n_targets'] = dict(heldout_spec=int((top1[top1.method == 'Proposed ICP inverse'].kind == 'heldout_spec').sum()),
                              grid=int((top1[top1.method == 'Proposed ICP inverse'].kind == 'grid').sum()))
cv['init'] = cv.method.str.split(':').str[1]
cv['steps'] = cv.method.str.split(':').str[2].astype(int)
N['amortisation_curve'] = cv.groupby(['init', 'steps']).agg(success=('success', 'mean'), time=('time_s', 'mean')).reset_index().to_dict(orient='records')
check('inv_n_spec', f"Table~\\ref{{tab:inv}} uses {N['inverse_n_targets']['heldout_spec']} specifications")
check('inv_n_grid', f"and {N['inverse_n_targets']['grid']} single-frequency targets from 4.2 to 6.8~GHz")
check('inv_data_only', f"succeed in at most {max(inv[m][k] for m in ('Direct inverse MLP', 'Conditional VAE (data only)') for k in ('spec_success', 'grid_success')):.0f}\\% of the cases")
g_only = inv['Ablation: ICP generator only (no refinement)']
check('inv_generator', f"The generator alone meets {g_only['spec_success']:.0f}\\% of the specifications")
check('inv_generator_time', f"of the specifications in {g_only['time']:.2f}~s", machine_dependent=True)
check('inv_icp', f"The complete ICP search meets {icp['spec_success']:.0f}\\% of the specifications")
check('inv_icp_time', f"returns five diverse designs per query in {icp['time']:.1f}~s", machine_dependent=True)
check('inv_calibrated', f"has the highest calibrated success, {icp['calib_success']:.0f}\\%, against "
                        f"{inv['Constrained surrogate optimisation (random starts)']['calib_success']:.0f}\\% for constrained optimisation "
                        f"from random starts and {inv['Tandem network (unconstrained)']['calib_success']:.0f}\\% for the unconstrained tandem network")
check('inv_optimism', f"raises the median optimism from {icp['optimism_median']:.1f} to "
                      f"{inv['Ablation: no robust/support constraints']['optimism_median']:.1f}~dB")
cat = pd.read_csv('catalog_coverage.csv')
n_high = int(cat[cat.target_GHz > 6.45].catalog_n_matched.max())
check('inv_catalogue', f"Retrieval from the simulated catalogue succeeds in {inv['Catalogue retrieval (training HFSS designs)']['spec_success']:.0f}\\% "
                       f"of the specifications, but above 6.5~GHz at most {WORDS.get(n_high, n_high)} designs meet")
c0g, c0r = cv[cv.method == 'curve:generator:0'], cv[cv.method == 'curve:random:0']
calib0 = lambda c: 100 * ((c.worst_inband_dB + kap * c.oracle_sd_at_f) <= -10).mean()
check('inv_amortisation', f"the search reaches {calib0(c0g):.0f}\\% calibrated success without any refinement, against "
                          f"{calib0(c0r):.0f}\\% for random initialisation")

# ============================== physics and sensitivity ==============================
pck = json.load(open('physics_check.json'))
N['physics_check'] = pck
check('physics_fL', f"$fL={pck['icp_fL_mean']:.1f}\\pm{pck['icp_fL_std']:.1f}$~GHz$\\cdot$mm of the driven patch, close to "
                    f"${pck['hfss_fL_mean']:.1f}\\pm{pck['hfss_fL_std']:.1f}$~GHz$\\cdot$mm")
sens = pd.read_csv('parameter_sensitivity.csv').set_index('parameter')
N['sensitivity_MHz_per_mm'] = sens.median_abs_dfr_dx_MHz_per_unit.to_dict()
check('sensitivity', f"which they change by about {sens.loc['Curve_Amp1', 'median_abs_dfr_dx_MHz_per_unit'] / 1000:.2f} and "
                     f"{sens.loc['L_Patch', 'median_abs_dfr_dx_MHz_per_unit'] / 1000:.2f}~GHz per millimetre")

# ============================== Table IV: final designs ==============================
FD = pd.read_csv('final_designs_top3.csv')
b1 = FD[FD.priority == 1].sort_values('target_GHz')
sym = [('Air_gap', '$G$'), ('W_Patch', '$W$'), ('L_Patch', '$L$'), ('WS', '$W_T$'), ('Curve_Amp1', '$C$'), ('W_Patch1', '$W_1$'),
       ('L_Patch1', '$L_1$'), ('Sx', '$S_x$'), ('Sy', '$S_y$')]
heads = [f'{v:.1f}' + ('$^\\dagger$' if c == 'band' else '') for v, c in zip(b1.target_GHz, b1.case)]
rows4 = [[s] + [f'{v:.2f}' for v in b1[p]] for p, s in sym]
rows4.append(['$|S_{11}(f^*)|$'] + [f'${a:.0f}{{\\pm}}{b:.0f}$' for a, b in zip(b1.pred_S11_at_centre_dB, b1.pred_S11_std_dB)])
rows4.append(['$f_L$ (GHz)'] + [half_up(v) for v in b1.pred_band_lo_GHz])
rows4.append(['$f_H$ (GHz)'] + [half_up(v) for v in b1.pred_band_hi_GHz])
t4 = pd.DataFrame(rows4, columns=['$f^*$ (GHz)'] + heads)
t4.to_csv(f'{OUT}/table4_final_designs.csv', index=False)
tex(t4, f'{OUT}/table4_final_designs.tex', 'l' + 'c' * len(heads))
FD.to_csv(f'{OUT}/table4_all_designs_top3.csv', index=False)
band = b1[b1.case == 'band'].iloc[0]
check('band_case', f"the predicted mean band of {half_up(band.pred_band_lo_GHz)}--{half_up(band.pred_band_hi_GHz)}~GHz covers the "
                   f"requirement, whereas the calibrated bound narrows it to {half_up(band.calibrated_band_lo_GHz)}--"
                   f"{half_up(band.calibrated_band_hi_GHz)}~GHz")

json.dump(N, open(f'{OUT}/paper_numbers.json', 'w'), indent=1, default=float)
json.dump(CHECK, open(f'{OUT}/paper_text_checks.json', 'w'), indent=1)
pd.set_option('display.width', 200)
print(t2.to_string(index=False)); print(t3.to_string(index=False)); print(t4.to_string(index=False))
print(f'wrote {OUT}/ ({len(CHECK)} text numbers in paper_text_checks.json)')
