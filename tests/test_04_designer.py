"""Inverse design: the NumPy designer (designer/), its constraints, its command line and Table IV."""
import csv
import os

import numpy as np
import pandas as pd
import pytest

from conftest import DESIGNER, REF, latex_table_rows, run_tool

PARAMS = ['Air_gap', 'W_Patch', 'L_Patch', 'Sx', 'Sy', 'W_Patch1', 'WS', 'Curve_Amp1', 'Extra']


@pytest.fixture(scope='module')
def band_case(model):
    return model.design(4.9, bw=1.0, K=3, seed=2027)          # 4.4-5.4 GHz, the band of the reported antenna


@pytest.fixture(scope='module')
def centre_case(model):
    return model.design(5.5, K=3, seed=2027)                   # widest band centred on 5.5 GHz


def table4():
    return pd.read_csv(os.path.join(REF, 'final_designs_top3.csv'))


def test_band_case_reproduces_table4(model, band_case):
    ref = table4()
    ref = ref[(ref.case == 'band') & (ref.priority == 1)][PARAMS].values[0]
    assert np.allclose(model.from_unit(band_case['U'][0]), ref, atol=1e-9)


def test_centre_case_reproduces_table4(model, centre_case):
    ref = table4()
    ref = ref[(ref.case == 'centre') & (ref.target_GHz == 5.5) & (ref.priority == 1)][PARAMS].values[0]
    assert np.allclose(model.from_unit(centre_case['U'][0]), ref, atol=1e-9)


def test_hard_constraints(model, band_case, centre_case):
    """Box bounds, the 0.01 mm HFSS grid and the data support hold for every returned design."""
    for res in (band_case, centre_case):
        X = model.from_unit(res['U'])
        assert (X >= model.LB - 1e-9).all() and (X <= model.UB + 1e-9).all()
        assert np.allclose(np.round(X / 0.01) * 0.01, X, atol=1e-9)
        assert res['ev']['in_support'].all()
        U = res['U']
        assert all(np.abs(U[i] - U[j]).mean() >= 0.08 - 1e-12 for i in range(len(U)) for j in range(i))   # diverse


def test_centre_case_is_feasible_with_calibrated_margin(centre_case):
    ev = centre_case['ev']
    assert ev['feasible'][0] and ev['worst_rob_inband'][0] <= -10
    assert abs(ev['f_res'][0] - 5.5) < 0.25


def test_search_is_deterministic(model):
    a = model.design(6.2, K=2, seed=7)
    b = model.design(6.2, K=2, seed=7)
    assert np.array_equal(a['U'], b['U'])


def test_return_loss_target_is_respected(model):
    res = model.design(5.5, bw=0.3, K=2, thr=-15.0)
    ev = res['ev']
    assert ev['thr'] == -15.0
    for i in range(len(res['U'])):
        if ev['feasible'][i]:
            assert ev['worst_rob_inband'][i] <= -15.0


def test_rejects_targets_outside_the_trained_band(model):
    with pytest.raises(ValueError):
        model.design(7.5)


def test_command_line_band_and_circuit(tmp_path):
    out = tmp_path / 'run'
    txt = run_tool('design_antenna.py', ['--band', '5', '6', '--rl', '10', '--top', '1', '--circuit', '--out', str(out)], cwd=tmp_path)
    assert 'EXTRAPOLATION' in txt                      # 1 GHz is wider than every simulated -10 dB band (0.975 GHz)
    rows = list(csv.DictReader(open(out / 'designs.csv')))
    assert len(rows) == 1 and rows[0]['required_S11_dB'] == '-10.0'
    for r in rows:
        assert float(r['Curve_pos']) == pytest.approx(0.5 * float(r['L_Patch']) + 2, abs=1e-6)
        assert float(r['L_Patch1']) == pytest.approx(float(r['W_Patch1']) + float(r['Extra']), abs=0.011)
        s1p = np.array([[float(v) for v in ln.split()] for ln in open(out / f"predicted_{r['design_id']}.s1p") if ln[0] not in '!#'])
        assert s1p.shape == (3001, 3) and (s1p[:, 1] <= 1).all()
        circ = open(out / f"rf_circuit_{r['design_id']}.txt").read()
        assert 'parallel RLC resonators' in circ and 'complex RMSE' in circ
    for f in ('hfss_samples.csv', 'predicted_S11.png'):
        assert (out / f).exists()


def test_shipped_examples_are_up_to_date(tmp_path):
    out = tmp_path / 'ex'
    run_tool('design_antenna.py', ['--band', '5', '6', '--rl', '15', '--top', '3', '--out', str(out)], cwd=tmp_path)
    new = pd.read_csv(out / 'designs.csv')
    ref = pd.read_csv(os.path.join(DESIGNER, 'examples', '5-6GHz_RL15', 'designs.csv'))
    assert new.drop(columns='design_id').equals(ref.drop(columns='design_id'))


def test_interactive_mode(tmp_path):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, os.path.join(DESIGNER, 'design_antenna.py'), '--top', '1', '--out', str(tmp_path / 'i')],
                       input='5.8\n10\nn\n\n', capture_output=True, text=True, cwd=tmp_path, timeout=600)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'i' / 'designs.csv').exists()


def test_forward_tool_on_simulated_designs(tmp_path):
    out = tmp_path / 'pred'
    run_tool('predict_s11.py', ['--csv', os.path.join(REF, '..', '..', 'data', 'hfss', 'samples.csv'), '--rows', '291', '72', '--out', str(out)], cwd=tmp_path)
    P = pd.read_csv(out / 'predictions.csv')
    assert list(P.design_id) == ['run_0291', 'run_0072']
    assert (P.MAE_vs_HFSS_dB < 2.0).all()
    assert (out / 'predicted_run_0291.s1p').exists() and (out / 'predicted_S11.png').exists()


def test_table4_in_paper_matches_results():
    tex = open(os.path.join(REF, '..', '..', 'paper', 'Paper.tex'), encoding='utf-8').read()
    paper = latex_table_rows(tex, 'tab:designs')
    gen = pd.read_csv(os.path.join(REF, 'tables', 'table4_final_designs.csv'), dtype=str).values.tolist()
    assert gen == paper
