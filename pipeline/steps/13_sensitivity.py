"""Step 13 - relevance of each dimension (paper, Section II-B: C and L dominate the resonance frequency).

(i) kriging ARD length scales; (ii) median |d f_r / d x| of the dominant resonance from the deployed PC-Net ensemble,
by central differences over the matched designs, in MHz per mm.

Input : gp_oracle.pkl, processed.npz, design_descriptors.csv, pcnet_oracle.pt
Output: parameter_sensitivity.csv
"""
import _bootstrap  # noqa: F401
import numpy as np, pickle, pandas as pd, torch
from common import *
from inverse import Ensemble
gp=pickle.load(open('gp_oracle.pkl','rb'))['gp']
ls=gp.kernel_.k1.k2.length_scale
pr=np.load('processed.npz'); U=to_unit(pr['X']); DS=pd.read_csv('design_descriptors.csv')
ens=Ensemble(torch.load('pcnet_oracle.pt'))
fg=np.round(np.arange(4,7.0001,0.001),3)
m=DS.matched.values; Um=U[m]; h=0.05
def fres(Uq):
    S=db(ens.complex_np(Uq,fg)).mean(0); return fg[np.argmin(S,1)]
base=fres(Um); sens=[]
for j in range(9):
    up=Um.copy(); up[:,j]=np.clip(up[:,j]+h,0,1); dn=Um.copy(); dn[:,j]=np.clip(dn[:,j]-h,0,1)
    d=(fres(up)-fres(dn))/((up[:,j]-dn[:,j])*(UB[j]-LB[j]))   # GHz per mm (or per unit)
    sens.append(np.median(np.abs(d))*1000)
T=pd.DataFrame(dict(parameter=PARAMS,range=[f'{a:g}-{b:g}' for a,b in zip(LB,UB)],kriging_ARD_length_scale=np.round(ls,3),
                    kriging_relevance=np.round(1/ls/np.max(1/ls),3),median_abs_dfr_dx_MHz_per_unit=np.round(sens,1)))
T=T.sort_values('kriging_relevance',ascending=False); T.to_csv('parameter_sensitivity.csv',index=False); print(T.to_string(index=False))
