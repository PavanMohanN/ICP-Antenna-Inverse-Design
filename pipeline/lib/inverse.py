"""Stage 3 - inverse, constrained and physically consistent design engine.

* Forward physics: ensemble of PC-Nets (passive/causal Foster decoders) -> mean and epistemic std of S11(dB).
* Inverse model : conditional generator  u = g(f*, BW, z)  trained *through* the frozen physics ensemble
                  (tandem learning) with explicit constraints:
                    - box constraints / HFSS bounds          : sigmoid output (hard, by construction)
                    - HFSS dependent variables               : Curve_pos, L_Patch1 computed exactly (hard)
                    - robust -10 dB band constraint          : mu + kappa*sigma <= -10 - margin over [f*-BW/2, f*+BW/2]
                    - data-support (trust-region) constraint : kNN distance to simulated designs <= d_max
                    - multi-solution diversity               : mode-seeking regulariser on latent z
* Test time    : sample many z, constrained refinement, snap to 0.01 mm HFSS grid, re-verify, rank, diversify.
"""
import numpy as np, torch, torch.nn as nn, time
from common import *
from models import PCNet, MLP

relu = torch.relu


class Ensemble:
    def __init__(self, states, h=128, depth=3):
        self.nets = []
        for s in states:
            n = PCNet(h, depth)
            n.load_state_dict(s)
            n.eval()
            for p in n.parameters():
                p.requires_grad_(False)
            self.nets.append(n)

    def db_stack(self, u, f):
        out = []
        for n in self.nets:
            (gr, gi), _ = n(u, f)
            out.append(10 * torch.log10(gr ** 2 + gi ** 2 + 1e-12))
        return torch.stack(out)  # (M,B,F)

    def complex_np(self, U, f):
        with torch.no_grad():
            u = torch.tensor(np.atleast_2d(U), dtype=torch.float32)
            ff = torch.tensor(f, dtype=torch.float32)
            out = [n(u, ff)[0] for n in self.nets]
        return np.stack([o[0].numpy() + 1j * o[1].numpy() for o in out])

    def circuit_np(self, U):
        with torch.no_grad():
            u = torch.tensor(np.atleast_2d(U), dtype=torch.float32)
            Ps = [n.params(u) for n in self.nets]
        return [{k: v.numpy() for k, v in P.items()} for P in Ps]


class Support:
    """Applicability-domain constraint: mean distance to k nearest simulated designs (unit cube)."""

    def __init__(self, Utr, k=3, q=0.95):
        self.U = torch.tensor(Utr, dtype=torch.float32)
        self.k = k
        Dm = torch.cdist(self.U, self.U)
        Dm.fill_diagonal_(1e9)
        d = Dm.topk(k, largest=False).values.mean(1)
        self.dmax = float(torch.quantile(d, q))

    def dist(self, u):
        return torch.cdist(u, self.U).topk(self.k, largest=False).values.mean(1)


T_BAND = torch.linspace(-0.5, 0.5, 9)


def band_freqs(fs, bw):
    return fs[:, None] + bw[:, None] * T_BAND[None, :]


def spec_loss(ens, u, fs, bw, kappa=1.0, margin=1.0, depth=-15.0, w_depth=0.02):
    S = ens.db_stack(u, band_freqs(fs, bw))
    mu, sd = S.mean(0), S.std(0)
    rob = mu + kappa * sd
    viol = relu(rob - (-10.0 - margin))
    c = mu[:, T_BAND.numel() // 2]
    return (viol ** 2).mean(1) + w_depth * relu(c - depth), viol


def cond_norm(fs, bw):
    return torch.stack([(fs - 5.5) / 1.5, (bw - 0.2) / 0.2], 1)


class Generator(nn.Module):
    def __init__(self, zdim=4, h=128, depth=3):
        super().__init__()
        self.zdim = zdim
        self.net = MLP(2 + zdim, 9, h, depth)

    def forward(self, fs, bw, z):
        return torch.sigmoid(self.net(torch.cat([cond_norm(fs, bw), z], 1)))


def sample_conditions(n, spec_pool, rng):
    """spec_pool: array (P,2) of (f_center, BW) achieved by simulated training designs."""
    n1 = int(0.75 * n)
    i = rng.integers(0, len(spec_pool), n1)
    fs = spec_pool[i, 0] + rng.normal(0, 0.05, n1)
    bw = spec_pool[i, 1] * rng.uniform(0.2, 1.0, n1)
    fs2 = rng.uniform(4.05, 6.95, n - n1)
    bw2 = rng.uniform(0.0, 0.08, n - n1)
    fs = np.clip(np.concatenate([fs, fs2]), 4.05, 6.95)
    bw = np.concatenate([bw, bw2])
    bw = np.minimum(bw, 2 * np.minimum(fs - 4.0, 7.0 - fs))
    return torch.tensor(fs, dtype=torch.float32), torch.tensor(bw, dtype=torch.float32)


def train_generator(ens, sup, spec_pool, iters=2500, batch=256, kappa=1.0, w_sup=50.0, w_div=0.05,
                    use_support=True, zdim=4, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = Generator(zdim)
    opt = torch.optim.Adam(gen.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-4)
    for it in range(iters):
        fs, bw = sample_conditions(batch, spec_pool, rng)
        z1 = torch.randn(batch, zdim) if zdim else torch.zeros(batch, 0)
        u1 = gen(fs, bw, z1)
        L, _ = spec_loss(ens, u1, fs, bw, kappa)
        loss = L.mean()
        if use_support:
            loss = loss + w_sup * (relu(sup.dist(u1) - sup.dmax) ** 2).mean()
        if zdim and w_div > 0:
            z2 = torch.randn(batch, zdim)
            u2 = gen(fs, bw, z2)
            L2, _ = spec_loss(ens, u2, fs, bw, kappa)
            loss = 0.5 * (loss + L2.mean() + (w_sup * (relu(sup.dist(u2) - sup.dmax) ** 2).mean() if use_support else 0))
            ratio = (u1 - u2).abs().mean(1) / ((z1 - z2).abs().mean(1) + 1e-5)
            loss = loss - w_div * torch.clamp(ratio, max=0.5).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % 500 == 0:
            print(it, float(loss))
    gen.eval()
    return gen


def refine(ens, sup, u0, fs, bw, steps=150, lr=0.05, kappa=1.0, w_sup=50.0, use_support=True):
    """Constrained refinement in logit space (box constraints remain hard)."""
    v = torch.logit(u0.clamp(1e-4, 1 - 1e-4)).detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        u = torch.sigmoid(v)
        L, _ = spec_loss(ens, u, fs, bw, kappa)
        loss = L.sum()
        if use_support:
            loss = loss + w_sup * (relu(sup.dist(u) - sup.dmax) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(v).detach()


def snap(u):
    x = from_unit(u)
    x = np.round(x / GRID_MM) * GRID_MM
    x = np.clip(x, LB, UB)
    return to_unit(x)


def evaluate_designs(ens, sup, U, fs, bw, kappa=1.0, fgrid=None):
    """Dense evaluation of candidate designs (numpy). Returns dict of arrays."""
    if fgrid is None:
        fgrid = np.round(np.arange(4.0, 7.0 + 1e-9, 0.005), 3)
    g = ens.complex_np(U, fgrid)
    S = db(g)
    mu, sd = S.mean(0), S.std(0)
    rob = mu + kappa * sd
    out = dict(mu=mu, sd=sd, rob=rob, fgrid=fgrid)
    j0 = np.argmin(np.abs(fgrid[None, :] - np.atleast_1d(fs)[:, None]), 1)
    lo_req = np.atleast_1d(fs) - np.atleast_1d(bw) / 2
    hi_req = np.atleast_1d(fs) + np.atleast_1d(bw) / 2
    inband = (fgrid[None, :] >= lo_req[:, None] - 1e-9) & (fgrid[None, :] <= hi_req[:, None] + 1e-9)
    out['worst_rob_inband'] = np.where(inband, rob, -np.inf).max(1)
    out['worst_mu_inband'] = np.where(inband, mu, -np.inf).max(1)
    out['mu_at_f'] = mu[np.arange(len(U)), j0]
    out['sd_at_f'] = sd[np.arange(len(U)), j0]
    rbw, mbw, f_res, rsym = [], [], [], []
    for i in range(len(U)):
        lo, hi = band_edges(fgrid, rob[i], fgrid[j0[i]])
        rbw.append(0.0 if np.isnan(lo) else hi - lo)
        rsym.append(0.0 if np.isnan(lo) else 2 * min(fgrid[j0[i]] - lo, hi - fgrid[j0[i]]))
        lo, hi = band_edges(fgrid, mu[i], fgrid[j0[i]])
        mbw.append(0.0 if np.isnan(lo) else hi - lo)
        # local resonance closest to target: minimum of mu within +-0.5 GHz
        w = np.abs(fgrid - fgrid[j0[i]]) <= 0.5
        f_res.append(fgrid[w][np.argmin(mu[i][w])])
    out['robust_BW'] = np.array(rbw)
    out['robust_symBW'] = np.array(rsym)
    out['mean_BW'] = np.array(mbw)
    out['f_res_local'] = np.array(f_res)
    with torch.no_grad():
        out['support_dist'] = sup.dist(torch.tensor(U, dtype=torch.float32)).numpy()
    out['in_support'] = out['support_dist'] <= sup.dmax + 1e-6
    out['feasible'] = (out['worst_rob_inband'] <= -10.0) & out['in_support']
    return out


B_TOL = 0.2  # GHz: centred robust band that tolerates the typical resonance-shift error of the surrogate


def rank_key(ev, fstar, bw):
    """Ranking rule shared by the benchmark and the executable:
    1) feasible (robust -10 dB over the requested band, inside the data support);
    2) robust -10 dB band centred on f* at least max(B_TOL, BW) wide (frequency-shift tolerance);
    3) deepest robust match over the requested band (largest margin)."""
    btol = min(max(B_TOL, bw), 2 * min(fstar - 4.0, 7.0 - fstar))
    tol = ev['robust_symBW'] >= btol - 1e-9
    return np.lexsort((ev['mu_at_f'], ev['worst_rob_inband'], ~tol, ~ev['feasible']))


def diverse_topk(order, U, K, dmin=0.08):
    chosen = []
    for i in order:
        if all(np.abs(U[i] - U[j]).mean() >= dmin for j in chosen):
            chosen.append(i)
        if len(chosen) == K:
            break
    return chosen


def propose(ens, sup, gen, fstar, bw, n=256, K=5, kappa=1.0, steps=150, use_refine=True, use_support=True,
            seed=0, init='mixed'):
    """Return up to K diverse designs for a single (f*, BW) request."""
    torch.manual_seed(seed)
    fs = torch.full((n,), float(fstar))
    bwt = torch.full((n,), float(bw))
    if init == 'generator':
        with torch.no_grad():
            u0 = gen(fs, bwt, torch.randn(n, gen.zdim))
    elif init == 'mixed':
        with torch.no_grad():
            u0 = torch.cat([gen(fs[:n // 2], bwt[:n // 2], torch.randn(n // 2, gen.zdim)), torch.rand(n - n // 2, 9)])
    else:
        u0 = torch.rand(n, 9)
    u = refine(ens, sup, u0, fs, bwt, steps=steps, kappa=kappa, use_support=use_support) if use_refine else u0
    Us = snap(u.numpy())
    ev = evaluate_designs(ens, sup, Us, np.full(n, fstar), np.full(n, bw), kappa)
    # rank: feasible first, then robust BW around f*, then depth at f*
    key = rank_key(ev, fstar, bw)
    top = diverse_topk(key, Us, K)
    return Us[top], {k: (v[top] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in ev.items()}
