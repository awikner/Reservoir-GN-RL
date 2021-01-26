from reservoir_rls_multires import *
import matplotlib.pyplot as plt
from lorenz63 import *
from scipy.signal import welch, periodogram, detrend
from sklearn.preprocessing import StandardScaler
import cma
import os

def min_func_wtruth(x, mask, base_data, f_s, true_external_data,\
    base_res, num_tests, num_nodes, pred_length, train_length, scale = True, 
    external_output = True):
    init_delay = 0
    data = base_data[init_delay:]
    external_data = true_external_data[init_delay:]
    if scale:
        SS = StandardScaler()
        external_data = SS.fit_transform(external_data.reshape(-1,1))
    funval = vt_min_function_norm_external(data,external_data, x, mask, base_res.Win, base_res.A, \
        num_tests = num_tests,  num_nodes = num_nodes, pred_length = pred_length, train_length = train_length,\
        external_output = external_output)
    return funval

data_length = 1000000
step = 0.05
f_s = 1/step

external_data = np.loadtxt('/lustre/awikner1/Reservoir-GN-RL/ornstein_uhlenbeck_data.csv', delimiter = ',')
lorenz_data_ou = np.loadtxt('/lustre/awikner1/Reservoir-GN-RL/lorenz_data_ou_step%0.2f.csv' %(step), delimiter = ',')

num_nodes = 360
num_tests = 200
train_length = 3000
sync_length = 500
pred_length = 500
res_seed = 1
base_res = reservoir(4,num_nodes,input_weight = 1, spectral_radius = 1, seed = res_seed) #Generate a reservoir
mask = ['input_weight', 'regularization', 'leakage', 'forget']
x0 = np.array([6,4,0,9])
min_func_base = lambda x: min_func_wtruth(x, mask, np.ascontiguousarray(lorenz_data_ou), f_s, external_data,\
    base_res, num_tests, num_nodes, pred_length, train_length)
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
"""
File where results are saved. IMPORTANT: Full covariance matrix is 
NOT saved, nor are the exact samples. If these need to be saved, one
will also have to download from github and make some edits. Again,
ask me.
"""
foldername = '/lustre/awikner1/Reservoir-GN-RL/cmaes_lorenz_ou_wexternout_scaled_res%d\\' % res_seed

if not os.path.exists(foldername):
    os.makedirs(foldername)
else:
    for root, dirs, files in os.walk(foldername):
        for file in files:
            os.remove(os.path.join(root, file))
opts.set('verb_filenameprefix',foldername)
results = cma.fmin(min_func_base, x0, sigma, options = opts) # Run the algorithm