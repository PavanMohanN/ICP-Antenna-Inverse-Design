"""Step 01 - data audit.

Reads the 500 HFSS Touchstone files and checks the design table before anything is learned.

Input : <ICP_DATA>/run_0000.s1p ... run_0499.s1p, samples.csv, runs.log
Output: raw.npz     frequencies f (31), complex S11 G (500 x 31), design table X (500 x 13), run ids
        audit.json  the checks below, in numbers

Checks
  * every file has the same 31-point grid (4-7 GHz, 100 MHz) and format (# GHz S MA R 50)
  * the 9 free variables in the file headers equal samples.csv
  * the two dependent HFSS variables are exact functions of the free ones:
        Curve_pos = 0.5 * L_Patch + 2,   L_Patch1 = W_Patch1 + Extra
  * H_sub and Curve_len1 are fixed
  * simulation time per design (runs.log)
"""
import glob
import json
import os
import re

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.signal import argrelmin
from sklearn.linear_model import LinearRegression

from settings import DATA_DIR

files = sorted(glob.glob(os.path.join(DATA_DIR, 'run_*.s1p')))
if len(files) == 0:
    raise SystemExit(f'No Touchstone files found in {DATA_DIR}. Set ICP_DATA or put the HFSS export in data/hfss/.')
rows, S = [], []
for fp in files:
    var, data, fmt = {}, [], None
    for ln in open(fp).read().splitlines():
        m = re.match(r'!\s+(\w+)\s*=\s*([-\d.eE+]+)(mm)?', ln)
        if m and 'File' not in ln:
            var[m.group(1)] = float(m.group(2))
        if ln.startswith('#'):
            fmt = ln
        if ln and not ln.startswith('!') and not ln.startswith('#'):
            data.append([float(v) for v in ln.split()])
    var['run_id'] = os.path.basename(fp)[:-4]
    var['fmt'] = fmt
    var['n'] = len(data)
    rows.append(var)
    S.append(np.array(data))
V = pd.DataFrame(rows).set_index('run_id')
S = np.stack(S)
f = S[0, :, 0]
assert all(np.allclose(s[:, 0], f) for s in S), 'frequency grids differ between files'
G = S[:, :, 1] * np.exp(1j * np.deg2rad(S[:, :, 2]))
db = 20 * np.log10(S[:, :, 1])

base = ['Air_gap', 'W_Patch', 'L_Patch', 'Sx', 'Sy', 'W_Patch1', 'WS', 'Curve_Amp1', 'Extra']
X = V.drop(columns=['fmt', 'n'])
print(f'{len(files)} designs | {len(f)} frequencies {f[0]:g}-{f[-1]:g} GHz | formats: {dict(V.fmt.value_counts())}')
print(X.describe().T[['min', 'max', 'mean']].round(3).to_string())

# dependent variables: linear fit on the free variables must be exact
derived = {}
for t, expect in [('Curve_pos', '0.5*L_Patch + 2'), ('L_Patch1', 'W_Patch1 + Extra')]:
    lr = LinearRegression().fit(X[base], X[t])
    resid = float(np.abs(X[t] - lr.predict(X[base])).max())
    coef = {k: round(float(c), 4) for k, c in zip(base, lr.coef_) if abs(c) > 1e-6}
    derived[t] = dict(expected=expect, fitted_coefficients=coef, intercept=round(float(lr.intercept_), 4), max_residual_mm=resid)
    print(f'{t} = {coef} + {lr.intercept_:.4f}   (expected {expect}; max residual {resid:.2e} mm)')

sm = pd.read_csv(os.path.join(DATA_DIR, 'samples.csv')).set_index('run_id')
csv_mismatch = float(np.abs(sm[base] - X.loc[sm.index, base]).max().max())
print('samples.csv vs Touchstone headers, max difference:', csv_mismatch)

cols = base + ['Curve_pos', 'L_Patch1', 'H_sub', 'Curve_len1']
np.savez('raw.npz', f=f, G=G, X=X[cols].values, cols=np.array(cols), ids=np.array(X.index))

# simulation time per design
runtime = None
log = os.path.join(DATA_DIR, 'runs.log')
if os.path.exists(log):
    t = np.array([int(v) for v in re.findall(r' OK (\d+)s', open(log).read())])
    runtime = dict(n=int(len(t)), mean_s=round(float(t.mean()), 1), min_s=int(t.min()), max_s=int(t.max()),
                   total_h=round(float(t.sum()) / 3600, 2))
    print(f"HFSS time per design: mean {runtime['mean_s']} s, total {runtime['total_h']} h")

mn = db.min(1)
nmin = np.bincount([len(argrelmin(d)[0]) for d in db])
audit = dict(n_designs=len(files), n_freq=len(f), f_GHz=[float(f[0]), float(f[-1])], formats=dict(V.fmt.value_counts()),
             free_variables=base, fixed={k: sorted(set(X[k])) for k in ('H_sub', 'Curve_len1')}, derived=derived,
             samples_csv_max_difference=csv_mismatch, hfss_runtime=runtime,
             designs_meeting_minus10dB_on_raw_grid=int((mn <= -10).sum()),
             min_on_band_edge=dict(low=int((db.argmin(1) == 0).sum()), high=int((db.argmin(1) == len(f) - 1).sum())),
             local_minima_histogram=[int(v) for v in nmin])
json.dump(audit, open('audit.json', 'w'), indent=1, default=str)
print('designs meeting -10 dB on the 31-point grid:', audit['designs_meeting_minus10dB_on_raw_grid'])
print('wrote raw.npz, audit.json')
