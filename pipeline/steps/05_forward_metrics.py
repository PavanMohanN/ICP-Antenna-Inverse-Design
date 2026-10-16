"""Step 05 - forward-model metrics from the out-of-fold predictions (paper, Table II and Section IV-B).

Input : processed.npz, design_descriptors.csv, fwd_oof_*.pkl
Output: fwd_cv_per_fold.csv, fwd_cv_mean.csv, fwd_cv_std.csv
Metrics per fold: dB MAE/RMSE, MAE near resonance (|S11| <= -6 dB), complex RMSE, -10 dB decision scores (precision,
recall, F1, AUC), passivity violations, resonance-frequency error, and the non-circular screening test
(precision@5: do the five top-ranked unseen designs really meet -10 dB at the queried frequency?).
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd, pickle, glob, json, os
from sklearn.metrics import roc_auc_score
from common import *

pr = np.load('processed.npz')
f, G, fd = pr['f'], pr['G'], pr['fd']
D = db(G)
DS = pd.read_csv('design_descriptors.csv')
F = folds()
OOF, times = {}, {}
for p in sorted(glob.glob('fwd_oof_*.pkl')):
    d = pickle.load(open(p, 'rb'))
    b = os.path.basename(p)
    fs = [int(c) for c in b.split('_f')[-1].split('.')[0]] if '_f' in b else list(range(5))
    for name, o in d['OOF'].items():
        if name not in OOF:
            OOF[name] = {k: (None if v is None else v.copy()) for k, v in o.items()}; times[name] = [np.nan] * 5
        else:
            for key, v in o.items():
                if v is None:
                    continue
                if OOF[name][key] is None:
                    OOF[name][key] = v.copy()
                else:
                    rows = ~np.isnan(np.abs(v).reshape(len(v), -1)).all(1)
                    OOF[name][key][rows] = v[rows]
        for j, k in enumerate(fs):
            times[name][k] = d['times'][name][j]


def fold_metrics(name, o, te):
    p = o['db31'][te]; t = D[te]
    r = {}
    r['dB MAE'] = np.abs(p - t).mean()
    r['dB RMSE'] = np.sqrt(((p - t) ** 2).mean())
    m = t <= -6
    r['dB MAE (resonant pts)'] = np.abs(p - t)[m].mean()
    r['R2 (dB)'] = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    c = o['c31'][te]
    r['complex RMSE'] = np.sqrt((np.abs(c - G[te]) ** 2).mean()) if np.isfinite(c).all() else np.nan
    yt, yp = (t <= -10).ravel(), (p <= -10).ravel()
    tp, fp, fn = (yt & yp).sum(), (~yt & yp).sum(), (yt & ~yp).sum()
    tn = (~yt & ~yp).sum()
    r['precision'] = tp / max(tp + fp, 1)
    r['recall'] = tp / max(tp + fn, 1)
    r['F1'] = 2 * tp / max(2 * tp + fp + fn, 1)
    r['balanced acc.'] = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    r['AUC'] = roc_auc_score(yt, -p.ravel())
    r['passivity violations (%)'] = 100 * (p > 0).mean()
    # resonance descriptors on matched designs (true dense min <= -10 dB)
    mt = DS.matched.values[te]
    if o['dbd'] is not None:
        pr_f = fd[np.argmin(o['dbd'][te], 1)]
        pr_s = o['dbd'][te].min(1)
    else:
        q = np.array([parabolic_min(f, s) for s in p]); pr_f, pr_s = q[:, 0], q[:, 1]
    r['f_res MAE (MHz)'] = 1000 * np.abs(pr_f - DS.f_res_GHz.values[te])[mt].mean()
    r['f_res within 50 MHz (%)'] = 100 * (np.abs(pr_f - DS.f_res_GHz.values[te])[mt] <= 0.05).mean()
    r['S11_min MAE (dB)'] = np.abs(pr_s - DS.S11_min_dB.values[te])[mt].mean()
    # non-circular screening test on unseen HFSS geometries
    P5, H1, base = [], [], []
    for j in range(31):
        truth = t[:, j] <= -10
        if truth.sum() == 0:
            continue
        score = p[:, j]
        top = np.argsort(score)[:5]
        P5.append(truth[top].mean()); H1.append(truth[top[0]]); base.append(truth.mean())
    r['screen precision@5 (%)'] = 100 * np.mean(P5)
    r['screen hit@1 (%)'] = 100 * np.mean(H1)
    r['screen base rate (%)'] = 100 * np.mean(base)
    return r


rows = []
for name, o in OOF.items():
    for k, te in enumerate(F):
        r = fold_metrics(name, o, te); r['model'] = name; r['fold'] = k
        r['train time per fold (s)'] = times[name][k]
        rows.append(r)
R = pd.DataFrame(rows)
order = ['ExtraTrees (dB)', 'Kriging/GP (dB)', 'MLP ensemble (dB)', 'MLP ensemble (Re/Im)', 'PC-Net (31-pt targets only)', 'PC-Net ensemble (proposed)']
order = [o for o in order if o in OOF]
mean = R.groupby('model').mean(numeric_only=True).loc[order].drop(columns='fold')
std = R.groupby('model').std(numeric_only=True).loc[order].drop(columns='fold')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(mean.T.round(3))
print(std.T.round(3))
R.to_csv('fwd_cv_per_fold.csv', index=False)
mean.to_csv('fwd_cv_mean.csv'); std.to_csv('fwd_cv_std.csv')
