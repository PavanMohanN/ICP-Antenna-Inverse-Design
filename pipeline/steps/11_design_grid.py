"""Step 11 - top-ranked design for every target from 4.1 to 6.9 GHz (paper, Fig. 5b), with the deployed designer.

Also records, for each target, the best existing HFSS design (catalogue coverage; paper, Section IV-C).

Input : deploy/model, processed.npz
Output: final_design_grid.csv   three designs per target with predicted bands and support distance
        catalog_coverage.csv    widest centred -10 dB band among the 500 simulated designs, per target
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd, json, time, sys, os
from icp_engine import ICPModel
from common import *
from settings import GRID_TARGETS_GHZ
mdl=ICPModel('deploy/model')
pr=np.load('processed.npz'); fd=pr['fd']; Dd=db(pr['Gd']); X=pr['X']
rows=[]
grid=np.array(GRID_TARGETS_GHZ)
t_all=[]
for fs in grid:
    t0=time.time(); res=mdl.design(float(fs),K=3,seed=2027); dt=time.time()-t0; t_all.append(dt)
    ev=res['ev']; Xs=mdl.from_unit(res['U'])
    for r in range(len(Xs)):
        d=dict(target_GHz=fs,rank=r+1,bw_req=res['bw_req'],time_s=dt,feasible=bool(ev['feasible'][r]),
               S11_at_f=ev['S11_at_f'][r],S11_sd=ev['S11_sd_at_f'][r],f_res=ev['f_res'][r],
               mean_lo=ev['mean_lo'][r],mean_hi=ev['mean_hi'][r],mean_BW=ev['mean_BW'][r],mean_symBW=ev['mean_symBW'][r],
               robust_lo=ev['robust_lo'][r],robust_hi=ev['robust_hi'][r],robust_BW=ev['robust_BW'][r],robust_symBW=ev['robust_symBW'][r],
               support_dist=ev['support_dist'][r])
        d.update({p:v for p,v in zip(PARAMS,Xs[r])})
        rows.append(d)
    print(fs, f'{dt:.1f}s', 'bw_req',res['bw_req'],'feas',ev['feasible'],'robSym',np.round(ev['robust_symBW'],3),'S@f',np.round(ev['S11_at_f'],1),flush=True)
R=pd.DataFrame(rows); R.to_csv('final_design_grid.csv',index=False)
# catalogue: best existing HFSS design (dense, physics-consistent) for each target
cat=[]
for fs in grid:
    j0=int(np.argmin(abs(fd-fs))); best=(0.0,-1,np.nan)
    for i in range(500):
        lo,hi=band_edges(fd,Dd[i],fd[j0])
        if np.isfinite(lo):
            sym=2*min(fs-lo,hi-fs)
            if sym>best[0]: best=(sym,i,Dd[i,j0])
    nfeas=int((Dd[:,j0]<=-10).sum())
    cat.append(dict(target_GHz=fs,catalog_best_symBW=best[0],catalog_best_id=best[1],catalog_S11_at_f=best[2],catalog_n_matched=nfeas))
pd.DataFrame(cat).to_csv('catalog_coverage.csv',index=False)
print('mean query time',np.mean(t_all))
