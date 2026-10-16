"""ICP inverse-design engine (NumPy only - no PyTorch needed at run time).

Inverse, Constrained and Physically consistent (ICP) design of the air-gap patch antenna:
    target frequency f* (and optional -10 dB bandwidth)  ->  9 independent HFSS variables (+2 derived, +2 fixed)

Physics : each ensemble member maps geometry to a Foster equivalent circuit
          Z(f) = R0 + j a f - j b/f + sum_k R_k / (1 + j Q_k (f/f_k - f_k/f)),  all elements >= 0,
          S11(f) = exp(-j 2 pi f tau) (Z - 1)/(Z + 1)   ->  |S11| <= 1 and causal for ANY geometry.
Inverse : conditional generator u = g(f*, BW, z) (trained through the frozen physics ensemble).
Constr. : HFSS box bounds (hard), dependent variables (hard), 0.01 mm grid (hard),
          robust spec  mean + kappa*std <= -10 dB over the requested band, data-support (trust region).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def _mlp(layers, x):
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = _silu(x)
    return x


class ICPModel:
    def __init__(self, model_dir=None):
        model_dir = model_dir or os.path.join(HERE, 'model')
        cfg = json.load(open(os.path.join(model_dir, 'config.json')))
        self.cfg = cfg
        self.params = cfg['params']
        self.LB, self.UB = np.array(cfg['lb']), np.array(cfg['ub'])
        self.KMAX = cfg['kmax']
        self.kappa = cfg['kappa']
        self.grid = cfg['grid_mm']
        W = np.load(os.path.join(model_dir, 'weights.npz'))
        self.nets = []
        for m in range(cfg['n_ensemble']):
            self.nets.append([(W[f'f{m}_W{i}'], W[f'f{m}_b{i}']) for i in range(cfg['n_layers_fwd'])])
        self.gen = [(W[f'g_W{i}'], W[f'g_b{i}']) for i in range(cfg['n_layers_gen'])]
        self.zdim = cfg['zdim']
        self.Usup = W['support_U']
        self.k_sup, self.dmax = cfg['support_k'], cfg['support_dmax']

    # ---------- physics-consistent forward model ----------
    def circuit(self, U, m):
        r = _mlp(self.nets[m], U)
        K = self.KMAX
        t = r[:, 4:].reshape(-1, K, 3)
        return dict(tau=0.6 * _sig(r[:, 0]), R0=_softplus(r[:, 1]) * 0.1, a=_softplus(r[:, 2]) * 0.2,
                    b=_softplus(r[:, 3]) * 2.0, fk=2.8 + 5.4 * _sig(t[..., 0]),
                    Qk=np.exp(1.0 + 5.0 * _sig(t[..., 1])), Rk=_softplus(t[..., 2]) * 5.0)

    @staticmethod
    def gamma_from_circuit(P, f):
        f = np.atleast_2d(f)
        Z = P['R0'][:, None] + 1j * (P['a'][:, None] * f - P['b'][:, None] / f)
        y = P['Qk'][:, :, None] * (f[:, None, :] / P['fk'][:, :, None] - P['fk'][:, :, None] / f[:, None, :])
        Z = Z + (P['Rk'][:, :, None] / (1 + 1j * y)).sum(1)
        return np.exp(-2j * np.pi * f * P['tau'][:, None]) * (Z - 1) / (Z + 1)

    def s11(self, U, f):
        """Complex S11 for every ensemble member: (M, N, F)."""
        U = np.atleast_2d(U)
        return np.stack([self.gamma_from_circuit(self.circuit(U, m), f) for m in range(len(self.nets))])

    def s11_db_stats(self, U, f):
        S = 20 * np.log10(np.maximum(np.abs(self.s11(U, f)), 1e-6))
        return S.mean(0), S.std(0)

    # ---------- constraints ----------
    def to_unit(self, x):
        return (np.asarray(x) - self.LB) / (self.UB - self.LB)

    def from_unit(self, u):
        return self.LB + np.asarray(u) * (self.UB - self.LB)

    def snap(self, U):
        x = np.clip(np.round(self.from_unit(U) / self.grid) * self.grid, self.LB, self.UB)
        return self.to_unit(x)

    def support_dist(self, U):
        d = np.sqrt(((U[:, None, :] - self.Usup[None, :, :]) ** 2).sum(-1))
        return np.sort(d, 1)[:, :self.k_sup].mean(1)

    @staticmethod
    def derived(x):
        x = np.atleast_2d(x)
        return 0.5 * x[:, 2] + 2.0, x[:, 5] + x[:, 8]  # Curve_pos, L_Patch1

    # ---------- inverse ----------
    def generate(self, fstar, bw, rng):
        """Learned inverse: one design per entry of the (fstar, bw) arrays."""
        fstar, bw = np.asarray(fstar, float), np.asarray(bw, float)
        c = np.stack([(fstar - 5.5) / 1.5, (bw - 0.2) / 0.2], 1)
        z = rng.standard_normal((len(fstar), self.zdim))
        return _sig(_mlp(self.gen, np.concatenate([c, z], 1)))

    def _objective(self, U, fstar, bw, thr=-10.0, margin=1.0, w_sup=50.0):
        """Constrained ICP objective for each candidate (fstar, bw are per-candidate arrays).
        thr: required |S11| in dB over the band (-10 dB = 10 dB return loss, -15 dB = 15 dB return loss)."""
        n_pts = 9                                         # band samples used in training and in the paper
        fb = fstar[:, None] + bw[:, None] * np.linspace(-0.5, 0.5, n_pts)[None, :]
        mu, sd = self.s11_db_stats(U, fb)
        viol = np.maximum(mu + self.kappa * sd - (thr - margin), 0.0)
        L = (viol ** 2).mean(1) + 0.02 * np.maximum(mu[:, n_pts // 2] - min(-15.0, thr - 5.0), 0.0)
        L += w_sup * np.maximum(self.support_dist(U) - self.dmax, 0.0) ** 2
        return L

    def refine(self, U0, fstar, bw, steps=60, lr=0.05, h=1e-3, thr=-10.0):
        """Constrained refinement in logit space (box constraints stay hard); forward-difference gradients + Adam."""
        v = np.log(np.clip(U0, 1e-4, 1 - 1e-4) / (1 - np.clip(U0, 1e-4, 1 - 1e-4)))
        m1, m2 = np.zeros_like(v), np.zeros_like(v)
        n, d = v.shape
        E = np.eye(d) * h
        fr = np.repeat(fstar, d + 1)
        br = np.repeat(bw, d + 1)
        for t in range(1, steps + 1):
            V = np.concatenate([v[:, None, :], v[:, None, :] + E[None]], 1).reshape(-1, d)
            L = self._objective(_sig(V), fr, br, thr=thr).reshape(n, d + 1)
            g = (L[:, 1:] - L[:, :1]) / h
            m1 = 0.9 * m1 + 0.1 * g
            m2 = 0.999 * m2 + 0.001 * g ** 2
            v = v - lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(m2 / (1 - 0.999 ** t)) + 1e-8)
        return _sig(v)

    def evaluate(self, U, fstar, bw, fgrid=None, thr=-10.0):
        fgrid = np.round(np.arange(4.0, 7.0 + 1e-9, 0.005), 3) if fgrid is None else fgrid
        mu, sd = self.s11_db_stats(U, fgrid)
        rob = mu + self.kappa * sd
        j0 = int(np.argmin(np.abs(fgrid - fstar)))
        inb = (fgrid >= fstar - bw / 2 - 1e-9) & (fgrid <= fstar + bw / 2 + 1e-9)
        inb[j0] = True
        out = dict(fgrid=fgrid, mu=mu, sd=sd, rob=rob)
        out['worst_rob_inband'] = rob[:, inb].max(1)
        out['worst_mu_inband'] = mu[:, inb].max(1)
        out['coverage_mean'] = (mu[:, inb] <= thr).mean(1)      # fraction of the requested band meeting thr
        out['coverage_robust'] = (rob[:, inb] <= thr).mean(1)
        out['thr'] = thr
        out['S11_at_f'] = mu[:, j0]
        out['S11_sd_at_f'] = sd[:, j0]
        res = []
        for i in range(len(U)):
            row = {}
            for tag, curve in (('mean', mu[i]), ('robust', rob[i])):
                lo, hi = _band(fgrid, curve, j0, thr)
                row[tag + '_lo'], row[tag + '_hi'] = lo, hi
                row[tag + '_BW'] = 0.0 if np.isnan(lo) else hi - lo
                row[tag + '_symBW'] = 0.0 if np.isnan(lo) else 2 * min(fstar - lo, hi - fstar)
            w = np.abs(fgrid - fstar) <= 0.5
            row['f_res'] = fgrid[w][np.argmin(mu[i][w])]
            row['S11_min'] = mu[i][w].min()
            res.append(row)
        for k in res[0]:
            out[k] = np.array([r[k] for r in res])
        out['support_dist'] = self.support_dist(U)
        out['in_support'] = out['support_dist'] <= self.dmax + 1e-9
        out['feasible'] = (out['worst_rob_inband'] <= thr) & out['in_support']
        return out

    def design(self, fstar, bw=None, K=5, n=160, steps=60, seed=2027, thr=-10.0):
        """Main entry point.  bw=None -> maximise the robust band centred on f* below thr (default -10 dB).
        thr: required |S11| in dB, e.g. -10 (10 dB return loss) or -15 (15 dB return loss).
        Candidates: half from the learned inverse generator, half uniform; several band levels searched in parallel."""
        if not (4.0 < fstar < 7.0):
            raise ValueError('f* must lie inside the trained band 4-7 GHz')
        rng = np.random.default_rng(seed)
        edge = 2 * min(fstar - 4.0, 7.0 - fstar)
        levels = [0.0, 0.15, 0.3, 0.45, 0.6] if bw is None else [bw]
        levels = [min(b, edge) for b in levels]
        per = n // len(levels)
        fs = np.full(per * len(levels), float(fstar))
        bws = np.repeat(levels, per)
        U0 = self.generate(fs, bws, rng)
        half = np.arange(len(fs)) % 2 == 1
        U0[half] = rng.random((half.sum(), 9))
        U = self.snap(self.refine(U0, fs, bws, steps=steps, thr=thr))
        breq = 0.0 if bw is None else bw
        ev = self.evaluate(U, fstar, breq, thr=thr)
        # 1) feasible  2) centred robust band >= max(0.2 GHz, BW) (tolerates resonance shift)  3) deepest robust match
        btol = min(max(0.2, breq), edge)
        tol = ev['robust_symBW'] >= btol - 1e-9
        key = np.lexsort((ev['S11_at_f'], ev['worst_rob_inband'], ~tol, ~ev['feasible']))
        chosen = []
        for i in key:
            if all(np.abs(U[i] - U[j]).mean() >= 0.08 for j in chosen):
                chosen.append(i)
            if len(chosen) == K:
                break
        return dict(bw_req=breq, maximise_bw=bw is None, thr=thr, U=U[chosen],
                    ev={k: (v[chosen] if isinstance(v, np.ndarray) and v.shape[:1] == (len(U),) else v) for k, v in ev.items()})


def _band(fgrid, curve, j0, thr=-10.0):
    if curve[j0] > thr:
        return np.nan, np.nan
    lo = j0
    while lo > 0 and curve[lo - 1] <= thr:
        lo -= 1
    hi = j0
    while hi < len(curve) - 1 and curve[hi + 1] <= thr:
        hi += 1
    return fgrid[lo], fgrid[hi]
