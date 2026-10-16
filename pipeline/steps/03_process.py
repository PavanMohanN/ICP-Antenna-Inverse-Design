"""Step 03 - physically consistent responses and response descriptors.

* Model order: for every design, the smallest K whose frequency hold-out error is within 10 % of the best K.
* The selected circuit (fitted to all 31 samples) gives a passive, causal response on a 1 MHz grid.
* Super-resolution benchmark: fit the 16 even samples, predict the 15 odd samples; the Foster model against
  linear, spline and PCHIP interpolants (paper, Section IV-A).
* Descriptors of every design: resonance f_r, minimum |S11|, -10 dB band, number of resonances.

Input : raw.npz, circuit_fits.pkl
Output: processed.npz            f, G (31 pt), X (500 x 9), ids, fd (1 MHz grid), Gd (500 x 3001), Ksel, P
        design_descriptors.csv   one row per design
        interp_benchmark_per_design.csv, interp_benchmark_summary.csv, frequency_coverage.csv, fit_quality.json
"""
import _bootstrap  # noqa: F401
import numpy as np, pickle, pandas as pd, json
from scipy.interpolate import CubicSpline, PchipInterpolator
import circuit as C
from common import *

f, G, X, ids = load()
D = db(G)
fits = pickle.load(open('circuit_fits.pkl', 'rb'))
ev, od = np.arange(0, 31, 2), np.arange(1, 31, 2)
KS = sorted(fits[0].keys())

# ---- model-order selection by frequency hold-out (smallest K within 10% of best hold-out error)
Ksel, P = [], []
for i in range(500):
    ho = np.array([fits[i][K]['ho'] for K in KS])
    k = KS[int(np.where(ho <= 1.10 * ho.min())[0][0])]
    Ksel.append(k)
    P.append(C.sort_tanks(fits[i][k]['p_all'], k))
Ksel = np.array(Ksel)

fd = np.round(np.arange(4.0, 7.0 + 1e-9, 0.001), 3)  # 1 MHz grid
Gd = np.stack([C.gamma_model(P[i], fd, Ksel[i]) for i in range(500)])
Dd = db(Gd)
Gfit31 = np.stack([C.gamma_model(P[i], f, Ksel[i]) for i in range(500)])

# ---- fit quality on all 31 HFSS points
fit_rmse = np.sqrt((np.abs(Gfit31 - G) ** 2).mean(1))
fit_dbmae = np.abs(db(Gfit31) - D).mean(1)
print('fit complex RMSE median/95%/max', np.percentile(fit_rmse, [50, 95, 100]))
print('fit dB MAE median/95%/max', np.percentile(fit_dbmae, [50, 95, 100]))
print('selected K counts', {k: int((Ksel == k).sum()) for k in KS})
print('max |Gamma| dense', np.abs(Gd).max())

# ---- interpolation benchmark on held-out odd frequencies (Table: physics-consistent super-resolution)
rows = []
for i in range(500):
    k = Ksel[i]
    pe = fits[i][k]['p_even']
    g_phys = C.gamma_model(pe, f[od], k)
    g_lin = np.interp(f[od], f[ev], G[i, ev].real) + 1j * np.interp(f[od], f[ev], G[i, ev].imag)
    g_spl = CubicSpline(f[ev], G[i, ev])(f[od])
    d_lin = np.interp(f[od], f[ev], D[i, ev])
    d_spl = CubicSpline(f[ev], D[i, ev])(f[od])
    d_pch = PchipInterpolator(f[ev], D[i, ev])(f[od])
    t = D[i, od]
    mask = t <= -6  # matched / resonant region
    for name, gdb, gc in [('Physics-consistent Foster model', db(g_phys), g_phys),
                          ('Linear (complex)', db(g_lin), g_lin),
                          ('Cubic spline (complex)', db(g_spl), g_spl),
                          ('Linear (dB)', d_lin, None), ('Cubic spline (dB)', d_spl, None), ('PCHIP (dB)', d_pch, None)]:
        r = dict(design=i, method=name, db_mae=np.abs(gdb - t).mean(), db_max=np.abs(gdb - t).max(),
                 db_mae_res=np.abs(gdb - t)[mask].mean() if mask.any() else np.nan,
                 c_rmse=np.sqrt((np.abs(gc - G[i, od]) ** 2).mean()) if gc is not None else np.nan,
                 feas_err=int(((gdb <= -10) != (t <= -10)).sum()),
                 passive_viol=int((gdb > 0).sum()))
        rows.append(r)
IB = pd.DataFrame(rows)
summ = IB.groupby('method').agg(dB_MAE=('db_mae', 'mean'), dB_MAE_resonant=('db_mae_res', 'mean'),
                                dB_maxerr_p95=('db_max', lambda s: np.percentile(s, 95)),
                                complex_RMSE=('c_rmse', 'mean'), threshold_flips=('feas_err', 'sum'),
                                passivity_violations=('passive_viol', 'sum')).sort_values('dB_MAE')
print(summ)

# ---- response descriptors from dense physics-consistent curves
desc = []
for i in range(500):
    s = Dd[i]
    j = int(np.argmin(s))
    lo, hi = band_edges(fd, s, fd[j])
    bwL, l2, h2 = longest_band(fd, s)
    # local minima (resonances) deeper than -6 dB
    lm = [q for q in range(1, len(s) - 1) if s[q] < s[q - 1] and s[q] <= s[q + 1] and s[q] <= -6]
    desc.append(dict(run_id=ids[i], K=Ksel[i], f_res_GHz=fd[j], S11_min_dB=s[j],
                     band_lo_GHz=lo, band_hi_GHz=hi, BW_main_GHz=(hi - lo) if np.isfinite(lo) else 0.0,
                     FBW_main_pct=100 * (hi - lo) / fd[j] if np.isfinite(lo) else 0.0,
                     BW_longest_GHz=bwL, n_resonances=len(lm), matched=bool(s[j] <= -10),
                     edge_touch=bool(np.isfinite(lo) and (lo <= 4.0 or hi >= 7.0)),
                     raw_min_dB=D[i].min(), raw_f_min_GHz=f[np.argmin(D[i])],
                     fit_cRMSE=fit_rmse[i], fit_dBMAE=fit_dbmae[i]))
DS = pd.DataFrame(desc)
for c, v in zip(PARAMS, X.T):
    DS[c] = v
der = derived(X)
DS['Curve_pos'] = der['Curve_pos']
DS['L_Patch1'] = der['L_Patch1']
print(DS[['f_res_GHz', 'S11_min_dB', 'BW_main_GHz', 'n_resonances']].describe())
print('matched (dense) designs', DS.matched.sum(), ' raw-grid matched', (D.min(1) <= -10).sum())
print('designs with dense min < raw min by >3 dB', ((DS.raw_min_dB - DS.S11_min_dB) > 3).sum())

# per-frequency coverage: fraction of designs meeting -10 dB at f (dense)
fc = np.round(np.arange(4.0, 7.0001, 0.05), 2)
cov = pd.DataFrame(dict(f_GHz=fc, n_matched=[int((Dd[:, np.argmin(abs(fd - x))] <= -10).sum()) for x in fc]))
cov['frac'] = cov.n_matched / 500

np.savez('processed.npz', f=f, G=G, X=X, ids=ids, fd=fd, Gd=Gd.astype(np.complex64), Ksel=Ksel,
         P=np.array([np.pad(p, (0, 4 + 3 * 5 - len(p))) for p in P]))
DS.to_csv('design_descriptors.csv', index=False)
IB.to_csv('interp_benchmark_per_design.csv', index=False)
summ.to_csv('interp_benchmark_summary.csv')
cov.to_csv('frequency_coverage.csv', index=False)
json.dump(dict(fit_cRMSE_median=float(np.median(fit_rmse)), fit_cRMSE_p95=float(np.percentile(fit_rmse, 95)),
               fit_dBMAE_median=float(np.median(fit_dbmae)), fit_dBMAE_p95=float(np.percentile(fit_dbmae, 95)),
               K_counts={int(k): int((Ksel == k).sum()) for k in KS}, max_abs_gamma_dense=float(np.abs(Gd).max())),
          open('fit_quality.json', 'w'), indent=1)
