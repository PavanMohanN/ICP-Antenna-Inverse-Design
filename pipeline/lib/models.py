"""Forward models: physics-consistent circuit-decoder network (PCNet) and black-box baselines."""
import numpy as np
import torch
import torch.nn as nn
from common import KMAX, decode_params, gamma_torch

torch.set_num_threads(2)


class MLP(nn.Module):
    def __init__(self, din, dout, h=128, depth=3, act=nn.SiLU):
        super().__init__()
        L = []
        d = din
        for _ in range(depth):
            L += [nn.Linear(d, h), act()]
            d = h
        L += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


class PCNet(nn.Module):
    """Geometry -> Foster equivalent circuit -> S11.  Passive & causal for every input."""

    def __init__(self, h=128, depth=3):
        super().__init__()
        self.body = MLP(9, 4 + 3 * KMAX, h, depth)
        # initialise tank resonances spread over band so the spectral loss sees every region
        with torch.no_grad():
            last = self.body.net[-1]
            last.weight.mul_(0.1)
            b = torch.zeros(4 + 3 * KMAX)
            b[0] = -1.5
            fk0 = torch.linspace(3.3, 7.7, KMAX)
            t = b[4:].view(KMAX, 3)
            t[:, 0] = torch.logit((fk0 - 2.8) / 5.4)
            t[:, 1] = 0.0
            t[:, 2] = 0.0
            last.bias.copy_(b)

    def params(self, x):
        return decode_params(self.body(x))

    def forward(self, x, f):
        P = self.params(x)
        return gamma_torch(P, f), P


def _loss_spec(gr, gi, Tr, Ti, w_db=0.02):
    lc = ((gr - Tr) ** 2 + (gi - Ti) ** 2).mean()
    m = torch.sqrt(gr ** 2 + gi ** 2 + 1e-8)
    mt = torch.sqrt(Tr ** 2 + Ti ** 2 + 1e-8)
    ldb = (20 * torch.log10(m.clamp_min(10 ** (-40 / 20))) - 20 * torch.log10(mt.clamp_min(10 ** (-40 / 20)))).abs().mean()
    return lc + w_db * ldb


def train_pcnet(Xtr, f_raw, Gtr, f_dense=None, Gd_tr=None, Xva=None, Gva=None, seed=0, epochs=4000,
                lr=3e-3, wd=1e-4, h=128, depth=3, w_dense=1.0, w_db=0.02, verbose=False):
    """Xtr: (N,9) unit-scaled; Gtr: (N,F) complex HFSS samples; Gd_tr: (N,Fd) dense physics-fitted targets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = PCNet(h, depth)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.02)
    X = torch.tensor(Xtr, dtype=torch.float32)
    fr = torch.tensor(f_raw, dtype=torch.float32)
    Tr, Ti = torch.tensor(Gtr.real, dtype=torch.float32), torch.tensor(Gtr.imag, dtype=torch.float32)
    if Gd_tr is not None:
        fd = torch.tensor(f_dense, dtype=torch.float32)
        Dr, Di = torch.tensor(Gd_tr.real, dtype=torch.float32), torch.tensor(Gd_tr.imag, dtype=torch.float32)
    best = (1e9, None)
    if Xva is not None:
        Xv = torch.tensor(Xva, dtype=torch.float32)
        Vr, Vi = torch.tensor(Gva.real, dtype=torch.float32), torch.tensor(Gva.imag, dtype=torch.float32)
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        (gr, gi), P = net(X, fr)
        loss = _loss_spec(gr, gi, Tr, Ti, w_db)
        if Gd_tr is not None and w_dense > 0:
            (dr, di) = gamma_torch(P, fd)
            loss = loss + w_dense * _loss_spec(dr, di, Dr, Di, w_db)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        if Xva is not None and (ep % 50 == 0 or ep == epochs - 1):
            net.eval()
            with torch.no_grad():
                (vr, vi), _ = net(Xv, fr)
                lv = ((vr - Vr) ** 2 + (vi - Vi) ** 2).mean().item()
            if lv < best[0]:
                best = (lv, {k: v.clone() for k, v in net.state_dict().items()}, ep)
            if verbose and ep % 500 == 0:
                print(ep, loss.item(), lv)
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    return net


def train_mlp(Xtr, Ytr, Xva=None, Yva=None, seed=0, epochs=3000, lr=3e-3, wd=1e-4, h=128, depth=3):
    torch.manual_seed(seed)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
    net = MLP(Xtr.shape[1], Ytr.shape[1], h, depth)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.02)
    X = torch.tensor(Xtr, dtype=torch.float32)
    Y = torch.tensor((Ytr - mu) / sd, dtype=torch.float32)
    best = (1e9, None)
    if Xva is not None:
        Xv = torch.tensor(Xva, dtype=torch.float32)
        Yv = torch.tensor((Yva - mu) / sd, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        loss = ((net(X) - Y) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if Xva is not None and (ep % 50 == 0 or ep == epochs - 1):
            with torch.no_grad():
                lv = ((net(Xv) - Yv) ** 2).mean().item()
            if lv < best[0]:
                best = (lv, {k: v.clone() for k, v in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()

    def predict(Xn):
        with torch.no_grad():
            return net(torch.tensor(Xn, dtype=torch.float32)).numpy() * sd + mu
    return predict
