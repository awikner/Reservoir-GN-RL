#!/homes/awikner1/.python-venvs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 24:00:00
#Launch on 20 cores distributed over as many nodes as needed
#SBATCH --ntasks=20
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

data_length = 1000000
step = 0.05
f_s = 1/step
scale = 0.01
slow_var = 48/28
lorenz_data_cosine = np.loadtxt('/lustre/awikner1/Reservoir-GN-RL/lorenz_data_cosine_step%0.2f.csv' %(step), delimiter = ',')
scaled_data = lorenz_data_cosine
scaled_data = np.ascontiguousarray(scaled_data)

num_nodes = 360
num_tests = 200
train_length = 3000
sync_length = 200
pred_length = 500
res_seed = 1
base_res = reservoir(3,num_nodes,input_weight = 1, spectral_radius = 1, seed = res_seed) #Generate a reservoir
mask = ['input_weight', 'regularization', 'leakage', 'spectral_radius', 'forget']
x0 = np.array([4.636771438402045, 5.6364128276072565, 5.673582356077067, 5.4])
forget = -2.*(np.log10(1.-0.99)+1.)
min_func_base = lambda x: vt_min_function_norm(np.ascontiguousarray(lorenz_data_cosine), \
    np.append(x, forget) , mask,base_res.Win, base_res.A, num_nodes, num_tests, \
    sync_length, train_length, pred_length)
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
foldername = '/lustre/awikner1/Reservoir-GN-RL/cmaes_lorenz_cosine_noextern_forget%f_res%d' % (forget, res_seed)
if not os.path.exists(foldername):
    os.makedirs(foldername)
else:
    for root, dirs, files in os.walk(foldername):
        for file in files:
            os.remove(os.path.join(root, file))
opts.set('verb_filenameprefix',foldername + '/')
results = cma.fmin(min_func_base, x0, sigma, options = opts) # Run the algorithm
