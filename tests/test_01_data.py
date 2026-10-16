"""Data: the 500 HFSS files, the design table and its constraints (pipeline step 01)."""
import glob
import os

import numpy as np
import pandas as pd
import pytest

from conftest import DATA, REF, run_step

FREE = ['Air_gap', 'W_Patch', 'L_Patch', 'Sx', 'Sy', 'W_Patch1', 'WS', 'Curve_Amp1', 'Extra']


def parse(fp):
    var, rows = {}, []
    for ln in open(fp):
        s = ln.strip()
        if s.startswith('!') and '=' in s and 'File' not in s:
            k, v = s[1:].split('=')
            var[k.strip()] = float(v.strip().replace('mm', ''))
        elif s and s[0] not in '!#':
            rows.append([float(x) for x in s.split()])
    return var, np.array(rows)


@pytest.fixture(scope='module')
def table():
    files = sorted(glob.glob(os.path.join(DATA, 'run_*.s1p')))
    out = [parse(f) for f in files]
    V = pd.DataFrame([v for v, _ in out], index=[os.path.basename(f)[:-4] for f in files])
    S = np.stack([d for _, d in out])
    return V, S


def test_500_designs_on_one_frequency_grid(table):
    V, S = table
    assert len(V) == 500
    assert S.shape == (500, 31, 3)
    assert np.allclose(S[:, :, 0], np.round(np.linspace(4, 7, 31), 6))


def test_hfss_data_is_passive(table):
    _, S = table
    assert (S[:, :, 1] <= 1.0 + 1e-9).all() and (S[:, :, 1] > 0).all()


def test_samples_csv_equals_touchstone_headers(table):
    V, _ = table
    sm = pd.read_csv(os.path.join(DATA, 'samples.csv')).set_index('run_id')
    assert np.abs(sm[FREE].values - V.loc[sm.index, FREE].values).max() == 0


def test_dependent_and_fixed_variables_are_exact(table):
    V, _ = table
    assert np.abs(V.Curve_pos - (0.5 * V.L_Patch + 2)).max() < 1e-9
    assert np.abs(V.L_Patch1 - (V.W_Patch1 + V.Extra)).max() < 1e-9
    assert set(V.H_sub) == {0.79} and set(V.Curve_len1) == {1.0}


def test_free_variables_inside_the_design_space(table):
    from common import LB, UB
    V, _ = table
    X = V[FREE].values
    assert (X >= LB - 1e-9).all() and (X <= UB + 1e-9).all()


def test_audit_step_reproduces_reference(tmp_path):
    """Step 01 run from scratch gives the same raw.npz and audit numbers as the paper run."""
    run_step('01_audit.py', tmp_path, [])
    new, ref = np.load(tmp_path / 'raw.npz', allow_pickle=True), np.load(os.path.join(REF, 'raw.npz'), allow_pickle=True)
    assert np.array_equal(new['G'], ref['G']) and np.array_equal(new['X'], ref['X']) and np.array_equal(new['f'], ref['f'])
    import json
    a = json.load(open(tmp_path / 'audit.json'))
    assert a['n_designs'] == 500 and a['designs_meeting_minus10dB_on_raw_grid'] == 265
    assert a['hfss_runtime']['mean_s'] == 160.6 and a['hfss_runtime']['total_h'] == 22.31
