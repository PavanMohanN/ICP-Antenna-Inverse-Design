"""Shared definitions: design space, data loading, folds, torch physics decoder, metrics."""
import numpy as np
import torch

PARAMS = ['Air_gap', 'W_Patch', 'L_Patch', 'Sx', 'Sy', 'W_Patch1', 'WS', 'Curve_Amp1', 'Extra']
LB = np.array([1.5, 10.0, 15.0, 0.2, 0.2, 8.0, 0.2, 0.5, 0.5])
UB = np.array([5.0, 23.0, 23.0, 5.0, 5.0, 17.0, 1.5, 4.0, 8.0])
FIXED = {'H_sub': 0.79, 'Curve_len1': 1.0}
GRID_MM = 0.01  # HFSS variable resolution used in the corpus
SEED = 2027


def derived(x):
    """HFSS dependent variables (exact in all 500 files)."""
    x = np.atleast_2d(x)
    return {'Curve_pos': 0.5 * x[:, 2] + 2.0, 'L_Patch1': x[:, 5] + x[:, 8]}


def to_unit(x):
    return (np.asarray(x) - LB) / (UB - LB)


def from_unit(u):
    return LB + np.asarray(u) * (UB - LB)


def load():
    d = np.load('raw.npz', allow_pickle=True)
    return d['f'], d['G'], d['X'][:, :9].astype(float), d['ids']


def folds(n=500, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


# ---------------- physics-consistent decoder (torch) -----------------
KMAX = 6


def decode_params(raw):
    """Map unconstrained network outputs to non-negative circuit elements.
    raw: (B, 4 + 3*KMAX).  Returns dict of tensors."""
    sp = torch.nn.functional.softplus
    tau = 0.6 * torch.sigmoid(raw[:, 0])               # ns, [0, 0.6]
    R0 = sp(raw[:, 1]) * 0.1
    a = sp(raw[:, 2]) * 0.2
    b = sp(raw[:, 3]) * 2.0
    t = raw[:, 4:].reshape(-1, KMAX, 3)
    fk = 2.8 + 5.4 * torch.sigmoid(t[..., 0])            # GHz, [2.8, 8.2]
    Qk = torch.exp(1.0 + 5.0 * torch.sigmoid(t[..., 1]))  # [2.7, 400]
    Rk = sp(t[..., 2]) * 5.0
    return dict(tau=tau, R0=R0, a=a, b=b, fk=fk, Qk=Qk, Rk=Rk)


def gamma_torch(P, f):
    """Complex S11 on frequency vector f (GHz tensor (F,) or per-sample (B,F)). Passive & causal by construction."""
    if f.dim() == 1:
        f = f[None, :]
    Zr = P['R0'][:, None] + 0 * f
    Zi = P['a'][:, None] * f - P['b'][:, None] / f
    fk, Qk, Rk = P['fk'][:, :, None], P['Qk'][:, :, None], P['Rk'][:, :, None]
    y = Qk * (f[:, None, :] / fk - fk / f[:, None, :])
    den = 1 + y ** 2
    Zr = Zr + (Rk / den).sum(1)
    Zi = Zi + (-Rk * y / den).sum(1)
    # Gamma = (Z-1)/(Z+1)
    nr, ni = Zr - 1, Zi
    dr, di = Zr + 1, Zi
    dd = dr ** 2 + di ** 2
    gr = (nr * dr + ni * di) / dd
    gi = (ni * dr - nr * di) / dd
    ph = -2 * np.pi * f * P['tau'][:, None]
    c, s = torch.cos(ph), torch.sin(ph)
    return gr * c - gi * s, gr * s + gi * c


def db(g):
    return 20 * np.log10(np.maximum(np.abs(g), 1e-6))


# ---------------- response descriptors -----------------

def band_edges(fd, sdb, f0, thr=-10.0):
    """Contiguous band (lo, hi) around f0 where S11 <= thr on dense grid; (nan,nan) if f0 not matched."""
    i0 = int(np.argmin(np.abs(fd - f0)))
    if sdb[i0] > thr:
        return np.nan, np.nan
    lo = i0
    while lo > 0 and sdb[lo - 1] <= thr:
        lo -= 1
    hi = i0
    while hi < len(fd) - 1 and sdb[hi + 1] <= thr:
        hi += 1
    return fd[lo], fd[hi]


def longest_band(fd, sdb, thr=-10.0):
    m = sdb <= thr
    best = (0.0, np.nan, np.nan)
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            bw = fd[j] - fd[i]
            if bw > best[0] or np.isnan(best[1]):
                best = (bw, fd[i], fd[j])
            i = j + 1
        else:
            i += 1
    return best


def parabolic_min(fr, s):
    """Sub-grid minimum of a sampled curve (3-point parabola)."""
    j = int(np.argmin(s))
    if 0 < j < len(s) - 1:
        y0, y1, y2 = s[j - 1], s[j], s[j + 1]
        den = y0 - 2 * y1 + y2
        if den > 1e-9:
            dx = 0.5 * (y0 - y2) / den
            return fr[j] + dx * (fr[1] - fr[0]), y1 - 0.25 * (y0 - y2) * dx
    return fr[j], s[j]


def load_oof(pattern='fwd_oof_*.pkl'):
    """Merge out-of-fold prediction pickles written by fold groups."""
    import glob, pickle, os
    OOF = {}
    for p in sorted(glob.glob(pattern)):
        d = pickle.load(open(p, 'rb'))
        for name, o in d['OOF'].items():
            if name not in OOF:
                OOF[name] = {k: (None if v is None else v.copy()) for k, v in o.items()}
                continue
            for key, v in o.items():
                if v is None:
                    continue
                if OOF[name][key] is None:
                    OOF[name][key] = v.copy()
                else:
                    rows = ~np.isnan(np.abs(v).reshape(len(v), -1)).all(1)
                    OOF[name][key][rows] = v[rows]
    return OOF
