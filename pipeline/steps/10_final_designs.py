"""Step 10 - final designs of Table IV, generated with the deployed NumPy designer (single source of truth).

Targets: 4.5, 5.0, 5.5, 5.8, 6.2 and 6.6 GHz with the widest band centred on f* (case 'centre'), and the band of the
reported antenna, 4.4-5.4 GHz (f* = 4.9 GHz, B = 1 GHz, case 'band'). Three diverse designs per target.

Input : deploy/model (step 07)
Output: final_designs_top3.csv          21 designs, rank 1-3 per target ('priority')
        design_runs/<target>/            the designer's usual output: designs.csv, hfss_samples.csv, .s1p, plot
"""
import os
import shutil

import _bootstrap  # noqa: F401
import pandas as pd

from design_antenna import run as design_run
from icp_engine import ICPModel

CENTRE_TARGETS = [4.5, 5.0, 5.5, 5.8, 6.2, 6.6]
BAND_CASE = (4.9, 1.0)          # 4.4-5.4 GHz, the band of the reported antenna
OUT = 'design_runs'

model = ICPModel('deploy/model')
shutil.rmtree(OUT, ignore_errors=True)
frames = []
cases = [(fs, None, 'centre', f'{fs:.2f}GHz') for fs in CENTRE_TARGETS]
cases.append((BAND_CASE[0], BAND_CASE[1], 'band', f'{BAND_CASE[0] - BAND_CASE[1] / 2:g}-{BAND_CASE[0] + BAND_CASE[1] / 2:g}GHz'))
for fs, bw, case, label in cases:
    rows = design_run(model, fs, bw, 3, os.path.join(OUT, label), seed=2027, rl=10.0, circuit=False, label=label)
    df = pd.DataFrame(rows)
    df.insert(1, 'case', case)
    df['requested_BW_GHz'] = 'max' if bw is None else bw
    df['priority'] = range(1, len(df) + 1)
    frames.append(df)
B = pd.concat(frames, ignore_index=True)
B.to_csv('final_designs_top3.csv', index=False)
print(B[['design_id', 'case', 'pred_S11_at_centre_dB', 'pred_S11_std_dB', 'pred_band_lo_GHz', 'pred_band_hi_GHz',
         'calibrated_BW_MHz', 'confidence']].to_string())
print('wrote final_designs_top3.csv and', OUT)
