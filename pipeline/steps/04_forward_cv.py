"""Step 04 - forward models, five-fold cross-validation by geometry (identical folds for every model).

Ground truth is the raw HFSS S11 of designs the model never saw. Usage (run.py calls it for you):
    python 04_forward_cv.py ET GP MLPdB MLPri      # baselines: Extra Trees, kriging, MLP on dB, MLP on Re/Im
    python 04_forward_cv.py PCraw PC               # PC-Net trained on the 31 samples only / with 10 MHz targets
    FOLDS=0,1,2 python 04_forward_cv.py PCraw PC   # a subset of folds (run.py splits the folds over --jobs processes)

Input : processed.npz
Output: fwd_oof_<models>_f<folds>.pkl   out-of-fold predictions (merged later by common.load_oof)
        pcnet_fold<k>.pt                 the five-member PC-Net of fold k (used by steps 09, 14 and 15)
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd, time, torch, json, sys, pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import roc_auc_score
from common import *
from models import *
from settings import CV_MEMBERS, ET_TREES, GP_RESTARTS, MLP_EPOCHS, PCNET_EPOCHS

torch.set_num_threads(1)
pr = np.load('processed.npz')
f, G, X, fd, Gd = pr['f'], pr['G'], pr['X'], pr['fd'], pr['Gd']
U = to_unit(X)
D = db(G)
F = folds()
M_ENS = CV_MEMBERS
fdn = np.round(np.arange(4.0, 7.0 + 1e-9, 0.01), 2)       # 10 MHz dense training targets
idx_dn = np.array([np.argmin(abs(fd - x)) for x in fdn])
Gdn = Gd[:, idx_dn]
fd_eval = fd  # 1 MHz for descriptors

OOF = {}  # model -> dict(db31=(500,31), c31=(500,31) or None, dbd=(500,len(fd)) or None, sd31)


def put(name, te, db31, c31=None, dbd=None, sd31=None):
    o = OOF.setdefault(name, dict(db31=np.full((500, 31), np.nan), c31=np.full((500, 31), np.nan + 0j),
                                  dbd=None, sd31=np.full((500, 31), np.nan)))
    o['db31'][te] = db31
    if c31 is not None:
        o['c31'][te] = c31
    if sd31 is not None:
        o['sd31'][te] = sd31
    if dbd is not None:
        if o['dbd'] is None:
            o['dbd'] = np.full((500, len(fd_eval)), np.nan, dtype=np.float32)
        o['dbd'][te] = dbd


def pcnet_predict(nets, Un, freq):
    with torch.no_grad():
        x = torch.tensor(Un, dtype=torch.float32)
        ff = torch.tensor(freq, dtype=torch.float32)
        out = [n(x, ff)[0] for n in nets]
    return np.stack([o[0].numpy() + 1j * o[1].numpy() for o in out])  # (M,B,F)


times = {}
which = sys.argv[1:] or ['ET', 'GP', 'MLPdB', 'MLPri', 'PCraw', 'PC']
import os
FSEL = [int(v) for v in os.environ.get('FOLDS', '0,1,2,3,4').split(',')]
for k, te in enumerate(F):
    if k not in FSEL:
        continue
    tr = np.setdiff1d(np.arange(500), te)
    rng = np.random.default_rng(k)
    print('fold', k, 'models', which, flush=True)
    if 'ET' in which:
        t0 = time.time()
        et = ExtraTreesRegressor(ET_TREES, min_samples_leaf=2, max_features=1.0, n_jobs=2, random_state=k).fit(U[tr], D[tr])
        put('ExtraTrees (dB)', te, et.predict(U[te]))
        times.setdefault('ExtraTrees (dB)', []).append(time.time() - t0)
    if 'GP' in which:
        t0 = time.time()
        ker = ConstantKernel(1.0) * Matern(length_scale=np.ones(9) * 0.5, nu=2.5) + WhiteKernel(1e-2)
        gp = GaussianProcessRegressor(ker, normalize_y=True, n_restarts_optimizer=GP_RESTARTS, random_state=k).fit(U[tr], D[tr])
        mu, sd = gp.predict(U[te], return_std=True)
        put('Kriging/GP (dB)', te, mu, sd31=sd if sd.ndim == 2 else None)
        times.setdefault('Kriging/GP (dB)', []).append(time.time() - t0)
    # inner split for early stopping of neural models
    va_all = [rng.choice(tr, len(tr) // 10, replace=False) for _ in range(M_ENS)]
    if 'MLPdB' in which:
        t0 = time.time()
        P_ = []
        for m in range(M_ENS):
            va = va_all[m]; tr2 = np.setdiff1d(tr, va)
            P_.append(train_mlp(U[tr2], D[tr2], U[va], D[va], seed=m, epochs=MLP_EPOCHS)(U[te]))
        P_ = np.stack(P_)
        put('MLP ensemble (dB)', te, P_.mean(0), sd31=P_.std(0))
        times.setdefault('MLP ensemble (dB)', []).append(time.time() - t0)
    if 'MLPri' in which:
        t0 = time.time()
        Y = np.concatenate([G.real, G.imag], 1)
        P_ = []
        for m in range(M_ENS):
            va = va_all[m]; tr2 = np.setdiff1d(tr, va)
            y = train_mlp(U[tr2], Y[tr2], U[va], Y[va], seed=m, epochs=MLP_EPOCHS)(U[te])
            P_.append(y[:, :31] + 1j * y[:, 31:])
        P_ = np.stack(P_)
        g = P_.mean(0)
        put('MLP ensemble (Re/Im)', te, db(g), c31=g, sd31=db(P_).std(0))
        times.setdefault('MLP ensemble (Re/Im)', []).append(time.time() - t0)
    for tag, name, use_dense in [('PCraw', 'PC-Net (31-pt targets only)', False), ('PC', 'PC-Net ensemble (proposed)', True)]:
        if tag not in which:
            continue
        t0 = time.time()
        nets = []
        for m in range(M_ENS):
            va = va_all[m]; tr2 = np.setdiff1d(tr, va)
            nets.append(train_pcnet(U[tr2], f, G[tr2], fdn, Gdn[tr2] if use_dense else None, U[va], G[va], seed=m, epochs=PCNET_EPOCHS))
        g31 = pcnet_predict(nets, U[te], f)
        gd = pcnet_predict(nets, U[te], fd_eval)
        put(name, te, db(g31).mean(0), c31=g31.mean(0), dbd=db(gd).mean(0).astype(np.float32), sd31=db(g31).std(0))
        times.setdefault(name, []).append(time.time() - t0)
        if tag == 'PC':
            torch.save([n.state_dict() for n in nets], f'pcnet_fold{k}.pt')
pickle.dump(dict(OOF=OOF, times=times), open('fwd_oof_' + '_'.join(which) + '_f' + ''.join(map(str, FSEL)) + '.pkl', 'wb'))
print('done', {k: np.sum(v) for k, v in times.items()})
