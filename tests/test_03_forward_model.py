"""Forward model: PC-Net ensemble, cross-validation metrics (Table II) and calibration (pipeline steps 04-07)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import DESIGNER, REF, latex_table_rows, run_step


def test_predictions_are_passive_for_any_geometry(model):
    """Passive by construction: |S11| <= 1 for random geometries, including ones outside the sampled box."""
    rng = np.random.default_rng(1)
    U = rng.uniform(-0.5, 1.5, (400, 9))
    S = model.s11(U, np.linspace(3.5, 7.5, 161))
    assert np.abs(S).max() <= 1.0 + 1e-12


def test_deployed_model_is_the_paper_model():
    """designer/model is exactly the model exported by the paper run (step 07c)."""
    a = np.load(os.path.join(DESIGNER, 'model', 'weights.npz'))
    b = np.load(os.path.join(REF, 'deploy', 'model', 'weights.npz'))
    assert sorted(a.files) == sorted(b.files)
    assert all(np.array_equal(a[k], b[k]) for k in a.files)
    ca = json.load(open(os.path.join(DESIGNER, 'model', 'config.json')))
    cal = json.load(open(os.path.join(REF, 'calibration.json')))
    assert ca['kappa'] == cal['kappa']


@pytest.mark.torch
def test_numpy_engine_reproduces_pytorch_ensemble(model):
    import torch
    from inverse import Ensemble
    ens = Ensemble(torch.load(os.path.join(REF, 'pcnet_final.pt')))
    U = np.random.default_rng(0).random((64, 9))
    f = np.linspace(4, 7, 31)
    assert np.abs(model.s11(U, f) - ens.complex_np(U, f)).max() < 1e-4      # float64 NumPy vs float32 PyTorch


def test_resonance_follows_patch_length(model):
    """Sanity physics: a longer driven patch lowers the resonance (half-wavelength patch)."""
    fg = np.linspace(4, 7, 601)
    x = model.from_unit(np.full(9, 0.5))
    x[7] = 1.0                                       # small meander C (main family)
    res = []
    for L in (16.0, 18.0, 20.0, 22.0):
        x[2] = L
        mu, _ = model.s11_db_stats(model.to_unit(x)[None], fg)
        res.append(fg[np.argmin(mu[0])])
    assert all(a > b for a, b in zip(res, res[1:])), res


def test_step05_reproduces_table2_metrics(tmp_path):
    inputs = ['processed.npz', 'design_descriptors.csv', 'fwd_oof_ET_GP_MLPdB_MLPri.pkl', 'fwd_oof_PCraw_PC_f012.pkl', 'fwd_oof_PCraw_PC_f34.pkl']
    run_step('05_forward_metrics.py', tmp_path, inputs)
    new = pd.read_csv(tmp_path / 'fwd_cv_mean.csv').set_index('model')
    ref = pd.read_csv(os.path.join(REF, 'fwd_cv_mean.csv')).set_index('model')
    cols = [c for c in ref.columns if 'time' not in c]
    assert np.allclose(new[cols].values, ref[cols].values, equal_nan=True)


def test_step06_reproduces_kappa(tmp_path):
    inputs = ['processed.npz', 'fwd_oof_ET_GP_MLPdB_MLPri.pkl', 'fwd_oof_PCraw_PC_f012.pkl', 'fwd_oof_PCraw_PC_f34.pkl']
    run_step('06_calibrate.py', tmp_path, inputs)
    new = json.load(open(tmp_path / 'calibration.json'))
    ref = json.load(open(os.path.join(REF, 'calibration.json')))
    assert new['kappa'] == pytest.approx(ref['kappa'], rel=1e-12)
    assert round(new['kappa'], 2) == 1.22 and round(100 * new['precision_robust_rule'], 1) == 87.5


def test_calibrated_bound_covers_84_percent():
    """mu + kappa*sigma is an empirical one-sided 84 % bound on the out-of-fold points near resonance."""
    from common import db, load_oof
    pr = np.load(os.path.join(REF, 'processed.npz'), allow_pickle=True)
    D = db(pr['G'])
    o = load_oof(os.path.join(REF, 'fwd_oof_*.pkl'))['PC-Net ensemble (proposed)']
    mu, sd = o['db31'], o['sd31']
    m = (D <= -5) | (mu <= -5)
    kappa = json.load(open(os.path.join(REF, 'calibration.json')))['kappa']
    assert np.mean(D[m] <= mu[m] + kappa * sd[m]) == pytest.approx(0.84, abs=0.005)


def test_pcnet_beats_every_baseline_on_unseen_geometries():
    M = pd.read_csv(os.path.join(REF, 'fwd_cv_mean.csv')).set_index('model')
    pc = M.loc['PC-Net ensemble (proposed)']
    base = M.drop(index=['PC-Net ensemble (proposed)', 'PC-Net (31-pt targets only)'])
    assert pc['dB MAE'] < base['dB MAE'].min()
    assert pc['f_res MAE (MHz)'] < base['f_res MAE (MHz)'].min()
    assert pc['F1'] > base['F1'].max()
    assert pc['screen precision@5 (%)'] > base['screen precision@5 (%)'].max()
    assert pc['passivity violations (%)'] == 0


def test_table2_in_paper_matches_results():
    tex = open(os.path.join(REF, '..', '..', 'paper', 'Paper.tex'), encoding='utf-8').read()
    paper = latex_table_rows(tex, 'tab:fwd')
    gen = pd.read_csv(os.path.join(REF, 'tables', 'table2_forward_models.csv'), dtype=str).values.tolist()
    assert gen == paper
