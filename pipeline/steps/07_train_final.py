"""Step 07 - models trained on all 500 designs, and export for the NumPy designer.

    python 07_train_final.py oracle              5-member PC-Net on all designs (seeds 100-104) -> pcnet_oracle.pt
                                                 It is both the deployed forward model (copied to pcnet_final.pt by
                                                 run.py) and the reference that scores the inverse benchmark.
    python 07_train_final.py generator           conditional inverse generator g(f*, B, z) trained through the frozen
                                                 deployed ensemble -> generator_final.pt
    python 07_train_final.py export deploy/model NumPy weights + config.json for designer/icp_engine.py, followed by a
                                                 NumPy-versus-PyTorch parity check
"""
import _bootstrap  # noqa: F401
import numpy as np, torch, json, time, sys, os, pandas as pd
from common import *
from models import train_pcnet
from inverse import Ensemble, Support, train_generator
from settings import FINAL_MEMBERS, GEN_ITERS, MODE, PCNET_EPOCHS

torch.set_num_threads(1)
pr = np.load('processed.npz')
f, G, X, fd, Gd = pr['f'], pr['G'], pr['X'], pr['fd'], pr['Gd']
U = to_unit(X)
DS = pd.read_csv('design_descriptors.csv')
fdn = np.round(np.arange(4.0, 7.0 + 1e-9, 0.01), 2)
Gdn = Gd[:, [int(np.argmin(abs(fd - x))) for x in fdn]]
what = sys.argv[1]
KAPPA = float(json.load(open('calibration.json'))['kappa']) if os.path.exists('calibration.json') else 1.0
M = FINAL_MEMBERS


def ens_all(seeds, h=128, depth=3):
    states = []
    for s in seeds:
        rng = np.random.default_rng(s)
        va = rng.choice(500, 50, replace=False); tr = np.setdiff1d(np.arange(500), va)
        t0 = time.time()
        net = train_pcnet(U[tr], f, G[tr], fdn, Gdn[tr], U[va], G[va], seed=s, h=h, depth=depth, epochs=PCNET_EPOCHS)
        states.append(net.state_dict()); print('member', s, f'{time.time()-t0:.0f}s', flush=True)
    return states


if what == 'oracle':
    torch.save(ens_all(range(100, 100 + M)), 'pcnet_oracle.pt')
if what == 'final':
    states = ens_all(range(M))
    torch.save(states, 'pcnet_final.pt')
if what == 'generator':
    states = torch.load('pcnet_final.pt')
    ens = Ensemble(states)
    sup = Support(U)
    m = DS.matched.values
    pool = np.stack([(DS.band_lo_GHz.values[m] + DS.band_hi_GHz.values[m]) / 2, DS.BW_main_GHz.values[m]], 1)
    gen = train_generator(ens, sup, pool, kappa=KAPPA, seed=0, iters=GEN_ITERS)
    torch.save(gen.state_dict(), 'generator_final.pt')
if what == 'export':
    from inverse import Generator
    states = torch.load('pcnet_final.pt')
    gen = Generator(); gen.load_state_dict(torch.load('generator_final.pt'))
    W = {}
    for mi, s in enumerate(states):
        lin = [k for k in s if k.endswith('.weight')]
        for i, k in enumerate(lin):
            W[f'f{mi}_W{i}'] = s[k].numpy().astype(np.float64)
            W[f'f{mi}_b{i}'] = s[k.replace('weight', 'bias')].numpy().astype(np.float64)
    gs = gen.state_dict()
    lin = [k for k in gs if k.endswith('.weight')]
    for i, k in enumerate(lin):
        W[f'g_W{i}'] = gs[k].numpy().astype(np.float64)
        W[f'g_b{i}'] = gs[k.replace('weight', 'bias')].numpy().astype(np.float64)
    sup = Support(U)
    W['support_U'] = U.astype(np.float64)
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, 'weights.npz'), **W)
    cfg = dict(params=PARAMS, lb=LB.tolist(), ub=UB.tolist(), kmax=KMAX, kappa=KAPPA, grid_mm=GRID_MM,
               n_ensemble=len(states), n_layers_fwd=len([k for k in states[0] if k.endswith('.weight')]),
               n_layers_gen=len(lin), zdim=gen.zdim, support_k=sup.k, support_dmax=sup.dmax,
               fixed=FIXED, derived={'Curve_pos': '0.5*L_Patch + 2', 'L_Patch1': 'W_Patch1 + Extra'},
               units='mm', band_GHz=[4.0, 7.0], trained_on='500 HFSS 2024.2 designs (WP_AI_V1_sweep)')
    cfg['pipeline_mode'] = MODE
    # widest contiguous band below each threshold among the 500 designs (used for the extrapolation warning)
    pr_ = np.load('processed.npz', allow_pickle=True)
    fd_, D_ = pr_['fd'], 20 * np.log10(np.abs(pr_['Gd']))

    def _widest(row, thr):
        m = np.r_[False, row <= thr, False].astype(int)
        s, e = np.where(np.diff(m) == 1)[0], np.where(np.diff(m) == -1)[0] - 1
        return max([fd_[b] - fd_[a] for a, b in zip(s, e)], default=0.0)
    cfg['widest_band_in_data_GHz'] = {str(t): round(float(max(_widest(r, t) for r in D_)), 3) for t in (-10, -12, -15, -20)}
    json.dump(cfg, open(os.path.join(out, 'config.json'), 'w'), indent=1)
    # parity check: NumPy designer engine versus PyTorch
    from icp_engine import ICPModel
    mdl = ICPModel(out)
    ens = Ensemble(states)
    Ut = np.random.default_rng(0).random((64, 9))
    a = mdl.s11(Ut, f)
    b = ens.complex_np(Ut, f)
    err = float(np.abs(a - b).max())
    print('NumPy-versus-PyTorch max |dS11| =', err)
    assert err < 1e-4, 'NumPy engine does not reproduce the PyTorch ensemble'
    print('exported to', out)
