"""Step 09 - inverse-design benchmark on held-out specifications (paper, Table III and Fig. 5a).

    python 09_inverse_benchmark.py 0 1 2     # folds to run (run.py splits the folds over --jobs processes)

Targets  : (i) specs actually achieved by never-seen HFSS designs of the test fold (f_c, BW of their -10 dB band)
           (ii) a uniform frequency grid 4.2...6.8 GHz with BW = 0 (single-frequency match)
Evaluator: independent reference surrogate (PC-Net ensemble, different seeds, trained on all 500 designs).
           Inverse pipelines for fold k only ever see the fold-k training designs.
Methods  : catalogue retrieval, direct inverse MLP, conditional VAE, unconstrained tandem network, constrained
           optimisation from random starts, the proposed ICP search, and two ablations. On the grid targets the
           calibrated success is also recorded against the number of refinement steps ('curve:<init>:<steps>').

Input : processed.npz, design_descriptors.csv, calibration.json, pcnet_fold<k>.pt, pcnet_oracle.pt, gp_oracle.pkl
Output: inv_eval_rows_<folds>.csv (one row per proposed design), support.json
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd, torch, time, pickle, sys, json
import torch.nn as nn
from common import *
from models import MLP
from inverse import *
import settings as ST

torch.set_num_threads(1)
pr = np.load('processed.npz')
f, G, X, fd, Gd = pr['f'], pr['G'], pr['X'], pr['fd'], pr['Gd']
U = to_unit(X)
DS = pd.read_csv('design_descriptors.csv')
Ddense = db(Gd)
F = folds()
oracle = Ensemble(torch.load('pcnet_oracle.pt'))
GPOR = pickle.load(open('gp_oracle.pkl', 'rb'))  # Kriging oracle on 10 MHz processed curves
KTOP = 5
NPROP, STEPS = ST.INV_NPROP, ST.INV_STEPS
FOLDS = [int(a) for a in sys.argv[1:]] or list(ST.INV_FOLDS)


# ---------------- data-only inverse baselines ----------------
def spec_table(idx):
    m = DS.matched.values[idx]
    lo, hi = DS.band_lo_GHz.values[idx][m], DS.band_hi_GHz.values[idx][m]
    return idx[m], np.stack([(lo + hi) / 2, hi - lo], 1)


def train_direct(Utr, S, seed=0, epochs=ST.INV_DIRECT_EPOCHS):
    torch.manual_seed(seed)
    net = MLP(2, 9, 64, 3)
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-4)
    c = cond_norm(torch.tensor(S[:, 0], dtype=torch.float32), torch.tensor(S[:, 1], dtype=torch.float32))
    y = torch.tensor(Utr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); l = ((torch.sigmoid(net(c)) - y) ** 2).mean(); l.backward(); opt.step()
    net.eval()
    return lambda fs, bw: torch.sigmoid(net(cond_norm(fs, bw))).detach()


class CVAE(nn.Module):
    def __init__(s, zd=4):
        super().__init__(); s.zd = zd
        s.enc = MLP(9 + 2, 2 * zd, 64, 2); s.dec = MLP(zd + 2, 9, 64, 2)

    def forward(s, u, c):
        h = s.enc(torch.cat([u, c], 1)); mu, lv = h[:, :s.zd], h[:, s.zd:]
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return torch.sigmoid(s.dec(torch.cat([z, c], 1))), mu, lv


def train_cvae(Utr, S, seed=0, epochs=ST.INV_CVAE_EPOCHS, beta=0.5):
    torch.manual_seed(seed)
    m = CVAE(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    c = cond_norm(torch.tensor(S[:, 0], dtype=torch.float32), torch.tensor(S[:, 1], dtype=torch.float32))
    y = torch.tensor(Utr, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); r, mu, lv = m(y, c)
        l = ((r - y) ** 2).sum(1).mean() + beta * (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
        l.backward(); opt.step()
    m.eval()
    return lambda fs, bw, z: torch.sigmoid(m.dec(torch.cat([z, cond_norm(fs, bw)], 1))).detach()


def catalog(tr_idx, fs, bw):
    """Best simulated training design for the spec (uses its own HFSS-derived dense curve)."""
    lo, hi = fs - bw / 2, fs + bw / 2
    w = (fd >= lo - 1e-9) & (fd <= hi + 1e-9)
    worst = Ddense[tr_idx][:, w].max(1)
    return tr_idx[np.argsort(worst)[:KTOP]]


def gp_worst(Us, fs, bw):
    fg = GPOR['f']
    p = GPOR['gp'].predict(Us)
    w = (fg >= fs - bw / 2 - 1e-9) & (fg <= fs + bw / 2 + 1e-9)
    if not w.any():
        w = np.abs(fg - fs) == np.abs(fg - fs).min()
    return p[:, w].max(1)


def score(Us, fs, bw, method, fold, tgt, kind, t):
    ev = evaluate_designs(oracle, SUP_ALL, Us, np.full(len(Us), fs), np.full(len(Us), bw), kappa=0.0)
    gw = gp_worst(Us, fs, bw)
    rows = []
    for r in range(len(Us)):
        rows.append(dict(method=method, fold=fold, target=tgt, kind=kind, f_star=fs, bw_req=bw, rank=r,
                         success=bool(ev['worst_mu_inband'][r] <= -10.0), worst_inband_dB=ev['worst_mu_inband'][r],
                         success_gp=bool(gw[r] <= -10.0), worst_inband_gp=gw[r],
                         S11_at_f=ev['mu_at_f'][r], f_res_err_MHz=1000 * abs(ev['f_res_local'][r] - fs),
                         BW_oracle=ev['mean_BW'][r], support_dist=ev['support_dist'][r],
                         oracle_sd_at_f=ev['sd_at_f'][r], time_s=t, **{p: v for p, v in zip(PARAMS, from_unit(Us[r]))}))
    return rows


SUP_ALL = Support(U)
json.dump({'dmax_all': SUP_ALL.dmax, 'k': SUP_ALL.k}, open('support.json', 'w'))
KAPPA = float(json.load(open('calibration.json'))['kappa'])
rows = []
for k in FOLDS:
    te = F[k]; tr = np.setdiff1d(np.arange(500), te)
    t0 = time.time()
    ens = Ensemble(torch.load(f'pcnet_fold{k}.pt'))
    sup = Support(U[tr])
    tr_m, S_tr = spec_table(tr)
    gen = train_generator(ens, sup, S_tr, seed=k, kappa=KAPPA, iters=ST.INV_GEN_ITERS)
    tandem = train_generator(ens, sup, S_tr, seed=k, use_support=False, kappa=0.0, zdim=0, w_div=0.0, iters=ST.INV_GEN_ITERS)
    direct = train_direct(U[tr_m], S_tr, seed=k)
    cvae = train_cvae(U[tr_m], S_tr, seed=k)
    print(f'fold {k}: models trained in {time.time()-t0:.0f}s', flush=True)
    te_m, S_te = spec_table(te)
    sel = np.random.default_rng(100 + k).choice(len(te_m), min(ST.INV_SPECS_PER_FOLD, len(te_m)), replace=False)
    targets = [('heldout_spec', te_m[i], S_te[i, 0], S_te[i, 1]) for i in sorted(sel)]
    targets += [('grid', -1, fs, 0.0) for fs in ST.INV_GRID_GHZ]
    for kind, tid, fs, bw in targets:
        fs_t = torch.tensor([fs], dtype=torch.float32); bw_t = torch.tensor([bw], dtype=torch.float32)
        # 1 catalog retrieval
        t1 = time.time(); Uc = U[catalog(tr, fs, bw)]
        rows += score(Uc, fs, bw, 'Catalogue retrieval (training HFSS designs)', k, tid, kind, time.time() - t1)
        # 2 direct inverse regression
        t1 = time.time(); Ud = snap(direct(fs_t, bw_t).numpy())
        rows += score(Ud, fs, bw, 'Direct inverse MLP', k, tid, kind, time.time() - t1)
        # 3 data-only CVAE (z = 0 then samples)
        t1 = time.time(); z = torch.randn(KTOP, 4); z[0] = 0
        Uv = snap(cvae(fs_t.repeat(KTOP), bw_t.repeat(KTOP), z).numpy())
        rows += score(Uv, fs, bw, 'Conditional VAE (data only)', k, tid, kind, time.time() - t1)
        # 4 unconstrained tandem (deterministic, no support, no robustness)
        t1 = time.time()
        with torch.no_grad():
            Ut = snap(tandem(fs_t, bw_t, torch.zeros(1, 0)).numpy())
        rows += score(Ut, fs, bw, 'Tandem network (unconstrained)', k, tid, kind, time.time() - t1)
        # 5 constrained surrogate optimisation from random starts (no learned inverse)
        t1 = time.time(); Uo, _ = propose(ens, sup, gen, fs, bw, n=NPROP, K=KTOP, init='random', steps=STEPS, kappa=KAPPA, seed=tid + 7)
        rows += score(Uo, fs, bw, 'Constrained surrogate optimisation (random starts)', k, tid, kind, time.time() - t1)
        # 6 proposed: learned inverse prior + constrained physics-consistent refinement
        t1 = time.time(); Up, _ = propose(ens, sup, gen, fs, bw, n=NPROP, K=KTOP, init='mixed', steps=STEPS, kappa=KAPPA, seed=tid + 7)
        rows += score(Up, fs, bw, 'Proposed ICP inverse', k, tid, kind, time.time() - t1)
        # ablations
        t1 = time.time(); Ua, _ = propose(ens, sup, gen, fs, bw, n=NPROP, K=KTOP, init='generator', use_refine=False, kappa=KAPPA, seed=tid + 7)
        rows += score(Ua, fs, bw, 'Ablation: ICP generator only (no refinement)', k, tid, kind, time.time() - t1)
        t1 = time.time(); Ub, _ = propose(ens, sup, gen, fs, bw, n=NPROP, K=KTOP, init='mixed', steps=STEPS, kappa=0.0, use_support=False, seed=tid + 7)
        rows += score(Ub, fs, bw, 'Ablation: no robust/support constraints', k, tid, kind, time.time() - t1)
        if kind == 'grid':
            for init in ('generator', 'random'):
                for st in ST.INV_CURVE_STEPS:
                    t1 = time.time()
                    Us_, _ = propose(ens, sup, gen, fs, bw, n=NPROP, K=1, init=init, steps=st, use_refine=st > 0, kappa=KAPPA, seed=tid + 7)
                    rows += score(Us_, fs, bw, f'curve:{init}:{st}', k, tid, kind, time.time() - t1)
    print(f'fold {k} done {time.time()-t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(f'inv_eval_rows_{"_".join(map(str, FOLDS))}.csv', index=False)
