"""Physically consistent representation: the passive Foster model and its fits (pipeline steps 02-03)."""
import importlib.util
import json
import os
import pickle

import numpy as np
import pytest

import circuit as C
from conftest import REF, STEPS

F_DENSE = np.linspace(3.0, 9.0, 4001)


def random_params(rng, K):
    lo, hi = C._bounds(K)
    return lo + rng.random(len(lo)) * (hi - lo)


def test_foster_model_is_passive_for_any_parameters():
    """Non-negative elements -> positive-real Z -> |S11| <= 1 at every frequency (paper, eq. 1)."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        K = int(rng.integers(1, 7))
        p = random_params(rng, K)
        assert (C.z_model(p, F_DENSE, K).real >= -1e-12).all()
        assert (np.abs(C.gamma_model(p, F_DENSE, K)) <= 1 + 1e-12).all()


def test_fit_recovers_a_known_circuit():
    p_true = np.array([0.15, 0.02, 0.12, 0.6, 4.6, 15.0, 2.0, 5.3, 8.0, 0.9, 6.4, 25.0, 3.0])
    f = np.linspace(4, 7, 31)
    G = C.gamma_model(p_true, f, 3)
    p, cost = C.fit_one(f, G, 3)
    assert np.sqrt(np.mean(np.abs(C.gamma_model(p, f, 3) - G) ** 2)) < 1e-4
    assert np.allclose(np.sort(C.sort_tanks(p, 3)[4::3]), [4.6, 5.3, 6.4], atol=0.02)


def test_processed_responses_are_passive_and_fit_the_hfss_samples(processed):
    assert np.abs(processed['Gd']).max() <= 1.0
    q = json.load(open(os.path.join(REF, 'fit_quality.json')))
    assert q['fit_cRMSE_median'] < 0.02 and q['fit_dBMAE_median'] < 0.5 and q['max_abs_gamma_dense'] <= 1.0
    # the stored 1 MHz curve is the selected circuit evaluated on the grid
    i = 101
    K = int(processed['Ksel'][i])
    g = C.gamma_model(processed['P'][i][:4 + 3 * K], processed['fd'], K)
    assert np.abs(g - processed['Gd'][i]).max() < 1e-5


def test_model_order_selection_rule(processed):
    """K = smallest order whose frequency hold-out error is within 10 % of the best order."""
    fits = pickle.load(open(os.path.join(REF, 'circuit_fits.pkl'), 'rb'))
    for i in range(500):
        ks = sorted(fits[i])
        ho = np.array([fits[i][k]['ho'] for k in ks])
        assert ks[int(np.where(ho <= 1.10 * ho.min())[0][0])] == processed['Ksel'][i]


def test_step02_refit_is_reproducible(monkeypatch, processed):
    """Re-fitting one design with the code of step 02 gives the stored circuit (as good a fit, identical in the paper environment)."""
    monkeypatch.chdir(REF)
    monkeypatch.setenv('ICP_MODE', 'full')
    monkeypatch.syspath_prepend(STEPS)
    spec = importlib.util.spec_from_file_location('fit02', os.path.join(STEPS, '02_fit_circuits.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fits = pickle.load(open(os.path.join(REF, 'circuit_fits.pkl'), 'rb'))
    i = 101
    K = int(processed['Ksel'][i])
    p = mod.fit_fast(mod.f, mod.G[i], K, i + 1000)
    f, G = processed['f'], processed['G'][i]
    rmse = lambda q: np.sqrt(np.mean(np.abs(C.gamma_model(q, f, K) - G) ** 2))
    # same environment as the paper: identical circuit; other library versions: an equally good fit
    assert rmse(p) <= 1.05 * rmse(fits[i][K]['p_all']) + 1e-5


def test_super_resolution_benchmark_matches_paper():
    """Section IV-A: the Foster model predicts held-out samples far better than interpolation and is always passive."""
    import pandas as pd
    S = pd.read_csv(os.path.join(REF, 'interp_benchmark_summary.csv')).set_index('method')
    fo = S.loc['Physics-consistent Foster model']
    others = S.drop(index='Physics-consistent Foster model')
    assert fo.dB_MAE < others.dB_MAE.min()
    assert fo.passivity_violations == 0
    assert fo.threshold_flips < others.threshold_flips.min()
    assert round(fo.dB_MAE, 2) == 0.11


@pytest.mark.torch
def test_torch_decoder_equals_numpy_model():
    """The differentiable decoder inside PC-Net computes the same S11 as the NumPy designer engine."""
    import torch
    from common import decode_params, gamma_torch
    from icp_engine import ICPModel
    torch.manual_seed(0)
    raw = torch.randn(64, 4 + 3 * 6) * 2
    P = decode_params(raw)
    f = torch.linspace(4, 7, 61)
    gr, gi = gamma_torch(P, f)
    Pn = {k: v.numpy().astype(np.float64) for k, v in P.items()}
    gn = ICPModel.gamma_from_circuit(Pn, f.numpy().astype(np.float64))
    assert np.abs(gn - (gr.numpy() + 1j * gi.numpy())).max() < 1e-5
    assert torch.sqrt(gr ** 2 + gi ** 2).max().item() <= 1 + 1e-6
