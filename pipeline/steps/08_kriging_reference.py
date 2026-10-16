"""Step 08 - kriging reference on the 10 MHz physically consistent responses (diagnostic only).

Used for (i) a second, model-family-independent check of the inverse benchmark (it detects < 7 % of the matched
samples, so the paper scores with the PC-Net reference instead) and (ii) the ARD relevance of each dimension (step 13).

Input : processed.npz
Output: gp_oracle.pkl
"""
import _bootstrap  # noqa: F401
import numpy as np, pickle, time
from settings import KRIG_RESTARTS, KRIG_STEP_GHZ
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from common import *
pr=np.load('processed.npz'); fd=pr['fd']; Gd=pr['Gd']; U=to_unit(pr['X'])
fg=np.round(np.arange(4.0,7.0+1e-9,KRIG_STEP_GHZ),2); idx=[int(np.argmin(abs(fd-x))) for x in fg]
Y=db(Gd[:,idx])
t0=time.time()
gp=GaussianProcessRegressor(ConstantKernel(1.0)*Matern(np.ones(9)*0.5,nu=2.5)+WhiteKernel(1e-2),normalize_y=True,n_restarts_optimizer=KRIG_RESTARTS,random_state=0).fit(U,Y)
print(gp.kernel_, time.time()-t0)
pickle.dump(dict(gp=gp,f=fg),open('gp_oracle.pkl','wb'))
