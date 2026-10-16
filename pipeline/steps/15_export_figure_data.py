"""Step 15 - export every array the paper figures need into figure_data/ (small, portable files).

Afterwards figures/make_figures.py needs only NumPy, matplotlib and these files; nothing has to be retrained to
restyle a figure:   python run.py figures --run <name>

Input : processed.npz, circuit_fits.pkl, fwd_oof_*.pkl, pcnet_fold<k>.pt, design_descriptors.csv, inv_eval_rows_*.csv,
        final_design_grid.csv, final_designs_top3.csv, calibration.json, physics_check.json, deploy/model
Output: figure_data/  fig_extraction.npz, fig_forward.npz, fig_inverse_*.csv/.npz/.json
"""
import _bootstrap  # noqa: F401
import json, os, pickle, sys, glob
import numpy as np, pandas as pd, torch
import circuit as C
from common import *
from inverse import Ensemble

OUT = sys.argv[1] if len(sys.argv) > 1 else 'figure_data'
os.makedirs(OUT, exist_ok=True)
pr = np.load('processed.npz', allow_pickle=True)
f, G, X, fd = pr['f'], pr['G'], pr['X'], pr['fd']
U = to_unit(X)
D = db(G)
DS = pd.read_csv('design_descriptors.csv')
F = folds()
fold_id = np.zeros(500, int)
for k, te in enumerate(F):
    fold_id[te] = k

# ---------------- Fig. 3: frequency hold-out test on one design ----------------
fits = pickle.load(open('circuit_fits.pkl', 'rb'))
i = 101                                   # design with the deepest sampled resonance (run_0101)
K = int(pr['Ksel'][i])
pe = fits[i][K]['p_even']                 # Foster model fitted to the even-indexed samples only
fdx = np.round(np.arange(4, 7.0001, 0.001), 3)
ev = np.arange(0, 31, 2)
np.savez_compressed(f'{OUT}/fig_extraction.npz', f=f, S11=G[i], even_idx=ev, odd_idx=np.arange(1, 31, 2),
                    f_dense=fdx, S11_foster_dense=C.gamma_model(pe, fdx, K),
                    S11dB_linear_interp=np.interp(fdx, f[ev], D[i, ev]), design=str(pr['ids'][i]), K=K)

# ---------------- Fig. 4: forward model on unseen geometries ----------------
O = load_oof()
pc, et, gp = O['PC-Net ensemble (proposed)'], O['ExtraTrees (dB)'], O['Kriging/GP (dB)']
picks = [291, 72, 322]                     # median-error matched designs in 4.3-4.9, 5.1-5.7, 6.0-6.7 GHz
fdn = np.round(np.arange(4, 7.0001, 0.005), 3)
mu, sd = [], []
for i in picks:
    ens = Ensemble(torch.load(f'pcnet_fold{fold_id[i]}.pt'))   # the fold model that never saw design i
    S = db(ens.complex_np(U[i:i + 1], fdn))[:, 0]
    mu.append(S.mean(0)); sd.append(S.std(0))
np.savez_compressed(f'{OUT}/fig_forward.npz', f=f, f_dense=fdn, picks=np.array(picks), design=pr['ids'][picks].astype(str),
                    hfss_dB=D[picks], pcnet_mu=np.array(mu), pcnet_sd=np.array(sd), extratrees_dB=et['db31'][picks],
                    # parity + screening (all 500 designs, out-of-fold predictions)
                    S11dB_true=D, pred_dB_pcnet=pc['db31'], pred_dB_kriging=gp['db31'], pred_dB_extratrees=et['db31'],
                    fold=fold_id, matched=DS.matched.values, fres_true=DS.f_res_GHz.values,
                    fres_pcnet=fd[np.argmin(pc['dbd'], 1)])

# ---------------- Fig. 5: inverse design ----------------
R = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('inv_eval_rows_*.csv'))])
cv = R[R.method.str.startswith('curve:')].copy()
cv['init'] = cv.method.str.split(':').str[1]
cv['steps'] = cv.method.str.split(':').str[2].astype(int)
cv[['init', 'steps', 'fold', 'f_star', 'worst_inband_dB', 'oracle_sd_at_f', 'success']].to_csv(
    f'{OUT}/fig_inverse_amortisation.csv', index=False)
grid = pd.read_csv('final_design_grid.csv')
grid[grid['rank'] == 1][['target_GHz', 'L_Patch', 'Curve_Amp1', 'Air_gap', 'robust_symBW', 'mean_BW']].rename(
    columns={'L_Patch': 'L_mm', 'Curve_Amp1': 'C_mm', 'Air_gap': 'G_mm'}).to_csv(f'{OUT}/fig_inverse_icp_grid.csv', index=False)
DS[DS.matched][['run_id', 'f_res_GHz', 'L_Patch', 'Curve_Amp1']].rename(
    columns={'L_Patch': 'L_mm', 'Curve_Amp1': 'C_mm'}).to_csv(f'{OUT}/fig_inverse_hfss_matched.csv', index=False)
from icp_engine import ICPModel
mdl = ICPModel('deploy/model')
B = pd.read_csv('final_designs_top3.csv'); b1 = B[(B.priority == 1) & (B.case == 'centre')]
P9 = ['Air_gap', 'W_Patch', 'L_Patch', 'Sx', 'Sy', 'W_Patch1', 'WS', 'Curve_Amp1', 'Extra']
fg = np.round(np.arange(4, 7.0001, 0.005), 3)
mu6, sd6 = mdl.s11_db_stats(mdl.to_unit(b1[P9].values.astype(float)), fg)
np.savez_compressed(f'{OUT}/fig_inverse_designs.npz', f=fg, targets=b1.target_GHz.values, mu=mu6, sd=sd6,
                    dims=b1[P9].values.astype(float), dim_names=np.array(P9))
cal = json.load(open('calibration.json')); pc_ = json.load(open('physics_check.json'))
json.dump(dict(kappa=cal['kappa'], fL_hfss_mean=pc_['hfss_fL_mean'], fL_hfss_std=pc_['hfss_fL_std'],
               fL_icp_mean=pc_['icp_fL_mean'], fL_icp_std=pc_['icp_fL_std'], family_threshold_C_mm=2.0,
               L_lower_bound_mm=15.0, note='fL statistics use designs with C < 2 mm'),
          open(f'{OUT}/fig_inverse_meta.json', 'w'), indent=1)
print('exported to', OUT, sorted(os.listdir(OUT)))
