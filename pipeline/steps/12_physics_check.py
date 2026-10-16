"""Step 12 - physical consistency of the inverse designs beyond passivity (paper, Section IV-C and Fig. 5b).

Half-wavelength scaling of the driven patch, f_r * L = const, for the designs of the main family (C < 2 mm):
compared between the matched HFSS designs and the top-ranked ICP designs of step 11. The law is never imposed.

Input : design_descriptors.csv, final_design_grid.csv
Output: physics_check.json
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd, json
DS=pd.read_csv('design_descriptors.csv'); G=pd.read_csv('final_design_grid.csv')
m=DS.matched & (DS.Curve_Amp1<2.0)
fl=DS.f_res_GHz[m]*DS.L_Patch[m]
r=np.corrcoef(DS.f_res_GHz[m],1/DS.L_Patch[m])[0,1]
g=G[(G['rank']==1)&(G.Curve_Amp1<2.0)]
fl2=g.target_GHz*g.L_Patch
r2=np.corrcoef(g.target_GHz,1/g.L_Patch)[0,1] if len(g)>2 else np.nan
out=dict(hfss_n=int(m.sum()),hfss_fL_mean=float(fl.mean()),hfss_fL_std=float(fl.std()),hfss_corr_f_invL=float(r),
         icp_n=int(len(g)),icp_fL_mean=float(fl2.mean()),icp_fL_std=float(fl2.std()),icp_corr_f_invL=float(r2))
json.dump(out,open('physics_check.json','w'),indent=1); print(json.dumps(out,indent=1))
