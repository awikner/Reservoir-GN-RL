from reservoir_rls_multires import *
import matplotlib.pyplot as plt
from lorenz63 import *
from scipy.signal import welch, periodogram
from sklearn.preprocessing import StandardScaler
import cma
import os

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
mask = ['input_weight', 'regularization', 'leakage', 'forget']
x0 = np.array([6,4,0,9])
min_func_base = lambda x: vt_min_function_norm(np.ascontiguousarray(lorenz_data_cosine), x, mask,\
    base_res.Win, base_res.A, num_nodes, num_tests, sync_length, train_length, pred_length)
sigma = 2

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
opts.set('maxiter', 20)
"""
File where results are saved. IMPORTANT: Full covariance matrix is 
NOT saved, nor are the exact samples. If these need to be saved, one
will also have to download from github and make some edits. Again,
ask me.
"""
foldername = '/lustre/awikner1/Reservoir-GN-RL/cmaes_lorenz_cosine_noextern_res%d\\' % res_seed
if not os.path.exists(foldername):
    os.makedirs(foldername)
else:
    for root, dirs, files in os.walk(foldername):
        for file in files:
            os.remove(os.path.join(root, file))
opts.set('verb_filenameprefix',foldername)
results = cma.fmin(min_func_base, x0, sigma, options = opts) # Run the algorithm