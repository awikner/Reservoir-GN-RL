#!/homes/awikner1/anaconda3/envs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 3:00:00
#Launch on 40 cores distributed over as many nodes as needed
#SBATCH --ntasks=40
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
import ray

def min_func_wtruth(x, mask, base_data, f_s, true_external_data,\
     base_res, num_tests, num_nodes, pred_length, train_length, \
     sync_length, scale = True, external_output = True, progress = False):
     init_delay = 0
     data = base_data[init_delay:]
     external_data = true_external_data[init_delay:]
     if scale:
         SS = StandardScaler()
         external_data = SS.fit_transform(external_data.reshape(-1,1))
     funval = vt_min_function_norm_external(data,external_data, x, mask, base_res.Win, base_res.A, \
         num_tests = num_tests,  num_nodes = num_nodes, pred_length = pred_length, train_length = train_length,\
         external_output = external_output, progress = progress, returnall = True)
     return funval

@ray.remote
def get_results_ray(itr, num_tests, forget):
    data_length = 100000
    step = 0.05
    f_s = 1/step
    scale = 0.01
    slow_var = 48/28
    r_t = lambda x: r_t_cosine(x)
    dx_dt = lambda x, time, r_t: dxdt_lorenz(x, time, r_t)
    lorenz_data_cosine = getLorenzData(data_length, r_t, dx_dt, sample_tau = step, seed = itr)
    times = np.arange(lorenz_data_cosine.shape[0])*step
    external_data = r_t(times)
    scaled_data = np.ascontiguousarray(lorenz_data_cosine)
    num_nodes = 360
    train_length = 3000
    sync_length = 200
    pred_length = 500
    res_seed = 1
    base_res = reservoir(4,num_nodes,input_weight = 1, spectral_radius = 1, seed = res_seed) #Generate a reservoir
    mask = ['input_weight', 'regularization', 'leakage', 'spectral_radius', 'forget']
    x0 = np.array([5.096342469902955, 9.810183021133627, 2.2652262750064813, 5.163584080988384])
    min_func_base = lambda x: min_func_wtruth(np.append(x, forget), mask=mask, \
        base_data = scaled_data, f_s=f_s, true_external_data = external_data,\
        base_res=base_res, num_tests=num_tests, num_nodes=num_nodes, \
        pred_length=pred_length, train_length=train_length, sync_length = sync_length)
    result = min_func_base(x0)
    return result

ray.init(num_cpus = 40)

num_iters = 5000
num_tests = 100
forget = -2.*(np.log10(1.-0.98)+1.)

results = [get_results_ray.remote(i, num_tests, forget) for i in range(num_iters)]
results = ray.get(results)
results_data = np.zeros((num_iters, num_tests))
for i in range(num_iters):
    results_data[i] = results[i]
np.savetxt('cosine_external_truth_results_forget%f.csv' % (forget), results_data, delimiter = ',')

