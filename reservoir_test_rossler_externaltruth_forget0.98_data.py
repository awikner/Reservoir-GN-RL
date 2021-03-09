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
    r_t = lambda x: r_t_const(x)
    dx_dt = lambda x, time, r_t: dxdt_lorenz_rossler(x, time, r_t, scale = scale, slow_var = slow_var)
    lorenz_data_rossler = getCoupledLorenzData(data_length, r_t, dx_dt, sample_tau = step, seed = itr)
    scaled_data = lorenz_data_rossler[:,:3]
    scaled_data = np.ascontiguousarray(scaled_data)
    external_data = lorenz_data_rossler[:,4]
    num_nodes = 360
    train_length = 3000
    sync_length = 200
    pred_length = 500
    res_seed = 1
    base_res = reservoir(4,num_nodes,input_weight = 1, spectral_radius = 1, seed = res_seed) #Generate a reservoir
    mask = ['input_weight', 'regularization', 'leakage', 'spectral_radius', 'forget']
    x0 = np.array([5.06141204882783, 9.959865329144177, 1.7826439773748008, 5.172836898878996])
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
np.savetxt('rossler_external_truth_results_forget%f.csv' % (forget), results_data, delimiter = ',')

