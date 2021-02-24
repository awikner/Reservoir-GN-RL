#!/homes/awikner1/.python-venvs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 24:00:00
#Launch on 20 cores distributed over as many nodes as needed
#SBATCH --ntasks=10
#SBATCH -N 1
#Assume need 6 GB/core (6144 MB/core)
#SBATCH --mem-per-cpu=6144
#SBATCH --mail-user=awikner1@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

import os, sys
sys.path.append('/lustre/awikner1/Reservoir-GN-RL/')

from reservoir_rls_multires import *
import matplotlib.pyplot as plt
from lorenz63 import *
from scipy.signal import welch, periodogram
from sklearn.preprocessing import StandardScaler
import cma
import functools


def min_func_wtruth(x, mask, base_data, f_s, true_external_data,\
    base_res, num_tests, num_nodes, pred_length, train_length, \
    sync_length, scale = True, external_output = True, progress = True):
    init_delay = 0
    data = base_data[init_delay:]
    external_data = true_external_data[init_delay:]
    if scale:
        SS = StandardScaler()
        external_data = SS.fit_transform(external_data.reshape(-1,1))
    funval = vt_min_function_norm_external(data,external_data, x, mask, base_res.Win, base_res.A, \
        num_tests = num_tests,  num_nodes = num_nodes, pred_length = pred_length, train_length = train_length,\
        sync_length = sync_length, external_output = external_output, progress = progress)
    return funval

data_length = 1000000
step = 0.05
f_s = 1/step
scale = 0.01
slow_var = 48/28

lorenz_data_cosine = np.loadtxt('/lustre/awikner1/Reservoir-GN-RL/lorenz_data_cosine_step%0.2f.csv' %(step), delimiter = ',')
times = np.arange(lorenz_data_cosine.shape[0])*step
external_data = r_t_cosine(times)

scaled_data = lorenz_data_cosine
scaled_data = np.ascontiguousarray(scaled_data)

num_nodes = 360
num_tests = 200
train_length = 3000
sync_length = 200
pred_length = 500
res_seed = 1
base_res = reservoir(4,num_nodes,input_weight = 1, spectral_radius = 1, seed = res_seed) #Generate a reservoir
mask = ['input_weight', 'regularization', 'leakage', 'spectral_radius','forget']
x0 = np.array([5.071980365336762, 5.544142385647819, 2.605518524451397, 5.4])
forget = -2.*(np.log10(1.-0.99)+1.)
min_func_base = lambda x: min_func_wtruth(np.append(x, forget), mask=mask, \
        base_data = scaled_data, f_s=f_s, true_external_data = external_data,\
        base_res=base_res, num_tests=num_tests, num_nodes=num_nodes, \
        pred_length=pred_length, train_length=train_length, sync_length = sync_length)
sigma = 1.25

opts = cma.CMAOptions()
opts.set('popsize',10*x0.size) # Set number of samples per generation
"""
Set bounds on parameters. IMPORTANT: The mean returned by cma-es is
the mean BEFORE the boundary function is applied, so the mean may not
lie in the domain set by bounds. To obtain the true sample mean requires
downloading the cma-es package from github and editing one of the
functions. Ask me if you need to do this.
"""
opts.set('bounds', [0,10])
opts.set('seed', 5) # Seed for the initial samples
"""
File where results are saved. IMPORTANT: Full covariance matrix is
NOT saved, nor are the exact samples. If these need to be saved, one
will also have to download from github and make some edits. Again,
ask me.
"""
foldername = '/lustre/awikner1/Reservoir-GN-RL/cmaes_lorenz_cosine_wtruthout_scaled_forget%f_res%d' % (forget, res_seed)
if not os.path.exists(foldername):
    os.makedirs(foldername)
else:
    for root, dirs, files in os.walk(foldername):
        for file in files:
            os.remove(os.path.join(root, file))
opts.set('verb_filenameprefix',foldername + '/')

results = cma.fmin(min_func_base, x0, sigma, options = opts) # Run the algorithm
