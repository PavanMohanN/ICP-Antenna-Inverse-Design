"""Step 06 - calibrate the ensemble spread on out-of-fold predictions.

kappa is the 84 % quantile of the standardised error (HFSS - mu)/sigma over the points near resonance, so that
mu + kappa*sigma is an empirical one-sided 84 % upper bound of |S11| in dB ("calibrated bound" in the paper and tool).

Input : processed.npz, fwd_oof_*.pkl
Output: calibration.json   kappa, coverage and the precision/recall of the mean and calibrated -10 dB rules
"""
import _bootstrap  # noqa: F401
import numpy as np, pickle, json, glob
from common import *
pr=np.load('processed.npz'); D=db(pr['G'])
O=load_oof()
o=O['PC-Net ensemble (proposed)']; mu,sd=o['db31'],o['sd31']
m=(D<=-5)|(mu<=-5)
r=((D-mu)/np.maximum(sd,1e-3))[m]
kappa=float(np.quantile(r,0.84))
cov_raw=float(np.mean(np.abs(D-mu)[m]<=sd[m]))
qs=np.linspace(0.05,0.95,19)
obs=[float(np.mean(D[m]<=mu[m]+np.quantile(r,q)*sd[m])) for q in qs]
# one-sided decision check on all points: predicted robust-feasible -> truly feasible?
rob=mu+kappa*sd
prec_rob=float(((rob<=-10)&(D<=-10)).sum()/max((rob<=-10).sum(),1)); rec_rob=float(((rob<=-10)&(D<=-10)).sum()/max((D<=-10).sum(),1))
prec_mu=float(((mu<=-10)&(D<=-10)).sum()/max((mu<=-10).sum(),1)); rec_mu=float(((mu<=-10)&(D<=-10)).sum()/max((D<=-10).sum(),1))
out=dict(kappa=kappa,n_points=int(m.sum()),coverage_mu_pm_sigma_raw=cov_raw,median_sigma_dB=float(np.median(sd[m])),
         precision_mean_rule=prec_mu,recall_mean_rule=rec_mu,precision_robust_rule=prec_rob,recall_robust_rule=rec_rob)
json.dump(out,open('calibration.json','w'),indent=1); print(json.dumps(out,indent=1))
