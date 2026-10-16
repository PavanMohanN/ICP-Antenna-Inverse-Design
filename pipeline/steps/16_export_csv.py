"""Step 16 - human-readable copies of the processed data (CSV), for use outside Python.

Output (in csv/):
    design_table.csv                     500 designs: 9 free + 2 derived + 2 fixed variables
    hfss_S11_dB_31pt.csv                 raw HFSS |S11| in dB at the 31 simulated frequencies
    foster_circuit_parameters.csv        selected order K and the normalised Foster elements of every design
    design_descriptors.csv               resonance, minimum |S11|, -10 dB band, number of resonances, fit quality
The complex 1 MHz responses of every design are in processed.npz (keys fd, Gd).
"""
import os

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from common import PARAMS

OUT = 'csv'
os.makedirs(OUT, exist_ok=True)
pr = np.load('processed.npz', allow_pickle=True)
ids = pr['ids'].astype(str)
X = pd.DataFrame(pr['X'], columns=PARAMS)
X.insert(0, 'run_id', ids)
X['Curve_pos'] = 0.5 * X.L_Patch + 2
X['L_Patch1'] = X.W_Patch1 + X.Extra
X['H_sub'], X['Curve_len1'] = 0.79, 1.0
X.to_csv(f'{OUT}/design_table.csv', index=False)
S = pd.DataFrame(20 * np.log10(np.abs(pr['G'])), columns=[f'S11dB_{v:.1f}GHz' for v in pr['f']])
pd.concat([X[['run_id']], S], axis=1).to_csv(f'{OUT}/hfss_S11_dB_31pt.csv', index=False)
rows = []
for i in range(len(ids)):
    K, p = int(pr['Ksel'][i]), pr['P'][i]
    r = dict(run_id=ids[i], K=K, tau_ns=p[0], R0=p[1], a=p[2], b=p[3])
    for k in range(K):
        r.update({f'f{k + 1}_GHz': p[4 + 3 * k], f'Q{k + 1}': p[5 + 3 * k], f'R{k + 1}': p[6 + 3 * k]})
    rows.append(r)
pd.DataFrame(rows).to_csv(f'{OUT}/foster_circuit_parameters.csv', index=False)
pd.read_csv('design_descriptors.csv').to_csv(f'{OUT}/design_descriptors.csv', index=False)
print('wrote', OUT, sorted(os.listdir(OUT)))
