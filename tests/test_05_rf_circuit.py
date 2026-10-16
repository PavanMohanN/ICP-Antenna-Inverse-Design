"""RF equivalent circuit of a predicted S11 (designer/rf_circuit.py, option --circuit)."""
import numpy as np
import pytest

pytest.importorskip('scipy')
import rf_circuit as RF  # noqa: E402


def test_element_values_are_consistent():
    """Each tank: f_k = 1/(2 pi sqrt(L C)) and Q = R sqrt(C/L), with L in nH and C in pF."""
    p = np.array([0.18, 0.0, 0.2, 3.0, 5.3, 6.5, 0.78, 7.5, 36.5, 9.65])
    c = RF.to_physical(p, 2)
    for t in c['tanks']:
        L, C = t['L_nH'] * 1e-9, t['C_pF'] * 1e-12
        assert 1 / (2 * np.pi * np.sqrt(L * C)) / 1e9 == pytest.approx(t['f_GHz'], rel=1e-9)
        assert t['R_ohm'] * np.sqrt(C / L) == pytest.approx(t['Q'], rel=1e-9)
    assert c['L0_nH'] == pytest.approx(50 * 0.2 / (2 * np.pi))
    assert c['C0_pF'] == pytest.approx(1000 / (2 * np.pi * 50 * 3.0))


def test_fit_recovers_a_synthetic_antenna():
    p_true = np.array([0.17, 0.0, 0.25, 4.0, 4.4, 18.0, 8.0, 5.3, 6.5, 0.8, 5.85, 7.5, 0.5, 7.5, 36.0, 9.0])
    f = np.round(np.arange(4.0, 7.0 + 1e-9, 0.01), 2)
    G = RF.gamma(p_true, f, 4)
    c = RF.fit_equivalent_circuit(f, G)
    assert c['rmse'] < 2e-3 and c['max_dB_err'] < 0.5
    fitted = np.array([t['f_GHz'] for t in c['tanks']])
    for fk in (5.3, 5.85):
        assert np.min(np.abs(fitted - fk)) < 0.05
    assert all(t['R_ohm'] >= 0 and t['Q'] > 0 for t in c['tanks']) and c['R0_ohm'] >= 0
    txt = RF.describe(c)
    assert 'feed delay tau' in txt and 'parallel RLC resonators' in txt


def test_circuit_of_a_designed_antenna_is_passive(model):
    res = model.design(5.5, K=1)
    f = np.round(np.arange(4.0, 7.0 + 1e-9, 0.01), 2)
    S = model.s11(res['U'], f).mean(0)[0]
    c = RF.fit_equivalent_circuit(f, S, orders=(4, 5))
    assert np.abs(RF.gamma(c['p'], np.linspace(1, 20, 2000), c['K'])).max() <= 1 + 1e-12
    assert c['rmse'] < 0.02
