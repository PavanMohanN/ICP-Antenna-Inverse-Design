"""
rf_circuit.py  -  RF equivalent circuit of a predicted (or measured) one-port S11.

Model (the same passive, causal Foster form used in the paper):
    S11(f) = exp(-j 2 pi f tau) * (Z - Z0) / (Z + Z0),   Z0 = 50 ohm
    Z(f)   = R0 + j 2 pi f L0 + 1/(j 2 pi f C0)  +  sum_k  [ R_k || L_k || C_k ]
Every element is >= 0, so the circuit is passive and causal.
tau is the round-trip delay of the feed/reference plane (a lossless 50 ohm line of one-way delay tau/2).

Requires numpy and scipy.
"""
import numpy as np
from scipy.optimize import least_squares

Z0 = 50.0


def _z(p, f, K):
    tau, R0, a, b = p[:4]
    Z = R0 + 1j * a * f - 1j * b / f
    for k in range(K):
        fk, Qk, Rk = p[4 + 3 * k: 7 + 3 * k]
        Z = Z + Rk / (1 + 1j * Qk * (f / fk - fk / f))
    return Z


def gamma(p, f, K):
    Z = _z(p, f, K)
    return np.exp(-2j * np.pi * f * p[0]) * (Z - 1) / (Z + 1)


def _bounds(K):
    lo = [0.0, 0.0, 0.0, 0.0] + [3.0, 0.5, 0.0] * K
    hi = [2.5, 5.0, 5.0, 200.0] + [8.5, 400.0, 60.0] * K
    return np.array(lo), np.array(hi)


def _resid(p, f, G, K):
    d = gamma(p, f, K) - G
    return np.concatenate([d.real, d.imag])


def _fit(f, G, K, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = _bounds(K)
    starts = []
    for tau in (0.0, 0.1, 0.2, 0.3):
        Gd = G * np.exp(2j * np.pi * f * tau)
        Zr = ((1 + Gd) / (1 - Gd + 1e-9)).real
        pk = [j for j in range(1, len(f) - 1) if Zr[j] >= Zr[j - 1] and Zr[j] >= Zr[j + 1]]
        pk = sorted(pk, key=lambda j: -Zr[j])[:K]
        fk = [f[j] for j in pk] + list(rng.uniform(3.2, 7.8, K - len(pk)))
        p0 = [tau, 0.05, 0.1, 0.5]
        for k in range(K):
            j = int(np.argmin(abs(f - fk[k])))
            p0 += [fk[k], 20.0, float(np.clip(Zr[j], 0.2, 50))]
        starts.append(np.clip(p0, lo + 1e-6, hi - 1e-6))
    for _ in range(5):
        p0 = [rng.uniform(0, .4), rng.uniform(0, .3), rng.uniform(0, 1), rng.uniform(0, 5)]
        for _k in range(K):
            p0 += [rng.uniform(3.2, 7.8), np.exp(rng.uniform(np.log(5), np.log(100))), np.exp(rng.uniform(np.log(.3), np.log(20)))]
        starts.append(np.clip(p0, lo + 1e-6, hi - 1e-6))
    best = None
    for p0 in starts:
        r = least_squares(_resid, p0, bounds=(lo, hi), args=(f, G, K), method='trf', x_scale='jac', max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    r = least_squares(_resid, best.x, bounds=(lo, hi), args=(f, G, K), method='trf', x_scale='jac', max_nfev=3000)
    return r.x if r.cost < best.cost else best.x


def fit_equivalent_circuit(f, G, orders=(2, 3, 4, 5), tol=1.10):
    """Fit the Foster circuit to complex S11 samples G at frequencies f (GHz).
    The smallest order whose complex RMSE is within `tol` of the best is kept.
    Returns a dict with the physical element values and the fit quality."""
    fits = {}
    for K in orders:
        p = _fit(f, G, K)
        fits[K] = (p, np.sqrt(np.mean(np.abs(gamma(p, f, K) - G) ** 2)))
    best = min(v[1] for v in fits.values())
    K = min(k for k, v in fits.items() if v[1] <= tol * best)
    p, rmse = fits[K]
    db_err = np.abs(20 * np.log10(np.abs(gamma(p, f, K))) - 20 * np.log10(np.abs(G)))
    return dict(K=K, p=p, rmse=rmse, max_dB_err=float(db_err.max()), **to_physical(p, K))


def to_physical(p, K):
    """Normalised Foster parameters -> physical element values (ohm, nH, pF)."""
    tau, R0, a, b = p[:4]
    out = dict(tau_ns=tau, R0_ohm=Z0 * R0, L0_nH=Z0 * a / (2 * np.pi),
               C0_pF=(1000.0 / (2 * np.pi * Z0 * b)) if b > 1e-6 else np.inf)
    tanks = []
    for k in range(K):
        fk, Qk, Rk = p[4 + 3 * k: 7 + 3 * k]
        R = Z0 * Rk
        tanks.append(dict(f_GHz=fk, Q=Qk, R_ohm=R,
                          L_nH=R / (2 * np.pi * fk * Qk) if R > 0 else 0.0,
                          C_pF=1000.0 * Qk / (2 * np.pi * fk * R) if R > 0 else np.inf))
    out['tanks'] = sorted(tanks, key=lambda t: t['f_GHz'])
    return out


def describe(c):
    """Readable table of a fitted circuit."""
    c0 = 'none (short)' if not np.isfinite(c['C0_pF']) or c['C0_pF'] > 1e4 else f"{c['C0_pF']:.3g} pF"
    lines = [f"  feed delay tau = {c['tau_ns'] * 1000:.0f} ps (round trip)",
             f"  series: R0 = {c['R0_ohm']:.2f} ohm, L0 = {c['L0_nH']:.3f} nH, C0 = {c0}",
             '  parallel RLC resonators (in series):',
             '     f_k (GHz)     Q      R (ohm)    L (nH)    C (pF)   in 4-7 GHz?']
    for t in c['tanks']:
        lines.append(f"     {t['f_GHz']:8.3f}  {t['Q']:7.1f}  {t['R_ohm']:9.1f}  {t['L_nH']:8.4f}  {t['C_pF']:8.3f}   "
                     f"{'yes' if 4.0 <= t['f_GHz'] <= 7.0 else 'no (shapes the band edge)'}")
    lines.append(f"  fit: K = {c['K']} resonators, complex RMSE = {c['rmse']:.4f}, max |S11| error = {c['max_dB_err']:.2f} dB")
    return '\n'.join(lines)
