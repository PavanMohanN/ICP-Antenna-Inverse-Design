"""Step 02 - passive, causal equivalent circuit (Foster model) of every HFSS response.

For each design and each order K in settings.FIT_ORDERS (2..5 resonators), a bounded multistart least-squares fit of
    S11(f) = exp(-j 2 pi f tau) (Z - 1)/(Z + 1),  Z = R0 + j a f - j b/f + sum_k R_k / (1 + j Q_k (f/f_k - f_k/f))
is computed twice:
    p_even : fitted to the 16 even-indexed samples only -> scored on the 15 odd samples (frequency hold-out, 'ho')
    p_all  : fitted to all 31 samples                    -> used downstream ('tr' = training error)
All elements are bounded below by zero, so every fitted circuit is passive and causal.

Input : raw.npz
Output: circuit_fits.pkl   {design index: {K: dict(p_even, ho, p_all, tr)}}

The longest step of the pipeline (about 55 min on 2 cores in full mode). It runs in parallel over
ICP_WORKERS processes; the result does not depend on the number of processes.
"""
import pickle
import time
from multiprocessing import Pool

import _bootstrap  # noqa: F401
import numpy as np
from scipy.optimize import least_squares

import circuit as C
from settings import FIT_NFEV, FIT_ORDERS, FIT_POLISH_NFEV, FIT_RANDOM_STARTS, FIT_TAU_STARTS, WORKERS

d = np.load('raw.npz', allow_pickle=True)
f, G = d['f'], d['G']
ev, od = np.arange(0, 31, 2), np.arange(1, 31, 2)


def fit_fast(ff, GG, K, seed):
    rng = np.random.default_rng(seed)
    lo, hi = C._bounds(K)
    best, starts = None, []
    for tau in FIT_TAU_STARTS:                      # seeds from the peaks of Re Z after removing a trial delay
        Gd = GG * np.exp(2j * np.pi * ff * tau)
        Z = (1 + Gd) / (1 - Gd + 1e-9)
        pk = C._peaks(Z.real)
        fk = [ff[j] for j in pk[:K]]
        while len(fk) < K:
            fk.append(rng.uniform(3.2, 7.8))
        p0 = [tau, 0.05, 0.1, 0.5]
        for k in range(K):
            j = np.argmin(abs(ff - fk[k]))
            p0 += [fk[k], 20.0, float(np.clip(Z.real[j], 0.2, 50))]
        starts.append(np.clip(p0, lo + 1e-6, hi - 1e-6))
    for _ in range(FIT_RANDOM_STARTS):
        p0 = [rng.uniform(0, .4), rng.uniform(0, .3), rng.uniform(0, 1), rng.uniform(0, 5)]
        for _k in range(K):
            p0 += [rng.uniform(3.2, 7.8), np.exp(rng.uniform(np.log(5), np.log(100))), np.exp(rng.uniform(np.log(.3), np.log(20)))]
        starts.append(np.clip(p0, lo + 1e-6, hi - 1e-6))
    w = np.ones_like(ff)
    for p0 in starts:
        r = least_squares(C._resid, p0, bounds=(lo, hi), args=(ff, GG, K, w), method='trf', x_scale='jac', max_nfev=FIT_NFEV)
        if best is None or r.cost < best.cost:
            best = r
    r = least_squares(C._resid, best.x, bounds=(lo, hi), args=(ff, GG, K, w), method='trf', x_scale='jac',
                      max_nfev=FIT_POLISH_NFEV, xtol=1e-12, ftol=1e-12)
    if r.cost < best.cost:
        best = r
    return best.x


def work(i):
    out = {}
    for K in FIT_ORDERS:
        pe = fit_fast(f[ev], G[i, ev], K, i)
        ho = np.sqrt(np.mean(abs(C.gamma_model(pe, f[od], K) - G[i, od]) ** 2))
        pa = fit_fast(f, G[i], K, i + 1000)
        tr = np.sqrt(np.mean(abs(C.gamma_model(pa, f, K) - G[i]) ** 2))
        out[K] = dict(p_even=pe, ho=ho, p_all=pa, tr=tr)
    return i, out


if __name__ == '__main__':
    n = len(G)
    t0 = time.time()
    res = {}
    print(f'fitting {n} designs x orders {FIT_ORDERS} on {WORKERS} processes', flush=True)
    with Pool(WORKERS) as P:
        for done, (i, o) in enumerate(P.imap_unordered(work, range(n)), 1):
            res[i] = o
            if done % 25 == 0 or done == n:
                el = time.time() - t0
                print(f'  {done:3d}/{n} designs  {el / 60:5.1f} min elapsed, about {el / done * (n - done) / 60:5.1f} min left', flush=True)
    pickle.dump(res, open('circuit_fits.pkl', 'wb'))
    print(f'wrote circuit_fits.pkl in {(time.time() - t0) / 60:.1f} min')
