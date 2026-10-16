"""Passive, causal equivalent-circuit (Foster) representation of S11.

Gamma(f) = exp(-j 2 pi f tau) * (Z(f) - 1) / (Z(f) + 1)
Z(f)     = R0 + j a f - j b / f + sum_k R_k / (1 + j Q_k (f/f_k - f_k/f))
All element values >= 0  ->  Z is positive-real  ->  |Gamma| <= 1 and causal for every
parameter vector (passivity/causality by construction). f in GHz, tau in ns, Z normalised to 50 ohm.
"""
import numpy as np
from scipy.optimize import least_squares

FMIN, FMAX = 4.0, 7.0


def z_model(p, f, K):
    tau, R0, a, b = p[:4]
    Z = R0 + 1j * a * f - 1j * b / f
    for k in range(K):
        fk, Qk, Rk = p[4 + 3 * k: 7 + 3 * k]
        Z = Z + Rk / (1 + 1j * Qk * (f / fk - fk / f))
    return Z


def gamma_model(p, f, K):
    Z = z_model(p, f, K)
    return np.exp(-2j * np.pi * f * p[0]) * (Z - 1) / (Z + 1)


def _resid(p, f, G, K, w):
    d = (gamma_model(p, f, K) - G) * w
    return np.concatenate([d.real, d.imag])


def _bounds(K):
    lo = [0.0, 0.0, 0.0, 0.0]
    hi = [2.5, 5.0, 5.0, 200.0]
    for _ in range(K):
        lo += [3.0, 0.5, 0.0]
        hi += [8.5, 400.0, 60.0]
    return np.array(lo), np.array(hi)


def _peaks(y):
    i = [j for j in range(1, len(y) - 1) if y[j] >= y[j - 1] and y[j] >= y[j + 1]]
    return sorted(i, key=lambda j: -y[j])


def fit_one(f, G, K, n_tau=17, rng=None, w=None):
    """Multi-start least-squares fit of a K-tank Foster model to complex S11 samples."""
    if rng is None:
        rng = np.random.default_rng(0)
    if w is None:
        w = np.ones_like(f)
    lo, hi = _bounds(K)
    best = None
    starts = []
    for tau in np.linspace(0.0, 1.6, n_tau):
        Gd = G * np.exp(2j * np.pi * f * tau)
        Z = (1 + Gd) / (1 - Gd + 1e-9)
        pk = _peaks(Z.real)
        fk = [f[j] for j in pk[:K]]
        while len(fk) < K:
            fk.append(rng.uniform(FMIN, FMAX))
        p0 = [tau, 0.05, 0.1, 0.5]
        for k in range(K):
            j = np.argmin(abs(f - fk[k]))
            p0 += [fk[k], 20.0, float(np.clip(Z.real[j], 0.2, 50))]
        starts.append(np.clip(p0, lo + 1e-6, hi - 1e-6))
    for _ in range(6):  # random restarts
        p0 = [rng.uniform(0, 1.6), rng.uniform(0, .3), rng.uniform(0, 1), rng.uniform(0, 5)]
        for k in range(K):
            p0 += [rng.uniform(FMIN, FMAX), np.exp(rng.uniform(np.log(5), np.log(100))), np.exp(rng.uniform(np.log(.3), np.log(20)))]
        starts.append(np.clip(p0, lo + 1e-6, hi - 1e-6))
    for p0 in starts:
        try:
            r = least_squares(_resid, p0, bounds=(lo, hi), args=(f, G, K, w), method='trf', x_scale='jac', max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    # polish
    r = least_squares(_resid, best.x, bounds=(lo, hi), args=(f, G, K, w), method='trf', x_scale='jac', max_nfev=3000, xtol=1e-12, ftol=1e-12)
    if r.cost < best.cost:
        best = r
    return best.x, best.cost


def sort_tanks(p, K):
    t = p[4:].reshape(K, 3)
    t = t[np.argsort(t[:, 0])]
    return np.concatenate([p[:4], t.ravel()])
