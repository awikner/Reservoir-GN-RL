#!/homes/awikner1/.python-venvs/reservoir-rls/bin/python -u
#Assume will be finished in no more than 18 hours
#SBATCH -t 12:00:00
#Launch on 6 cores distributed over as many nodes as needed
#SBATCH --ntasks=6
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

def phase_min_func(delay, truth, filtered):
    delay = round(delay)
    truth_delayed = truth[:truth.shape[0]-delay]
    filtered_delayed = filtered[delay:]
    val = -np.mean(truth_delayed * filtered_delayed)
    return val
def min_func_wphase(x, mask, base_data, f_s, true_external_data,\
    base_res, num_tests, num_nodes, pred_length, train_length, scale = True, 
    external_output = True, evenspace = False, returnall = False):
    init_delay = 10000
    max_freq = 0.15
    min_freq = 0.001
    Wn_xy = x[0]/10*(max_freq-min_freq)+min_freq
    Wn_z = Wn_xy
    data_split = separate_lorenz_2scale(base_data, f_s, Wn_xy, Wn_z, filter_order = 10)
    base_external_data = data_split[init_delay:,-1]
    num_delays = 2000
    z_centered = base_external_data - np.mean(base_external_data)
    min_func   = lambda delay: phase_min_func(delay, true_external_data[init_delay:], z_centered)
    func_vals = np.zeros(num_delays)
    for i in range(num_delays):
        func_vals[i] = min_func(i)
    min_delay = np.argmin(func_vals)
    data = base_data[init_delay:base_data.shape[0]-min_delay]
    external_data = base_external_data[min_delay:]
    if scale:
        SS = StandardScaler()
        external_data = SS.fit_transform(external_data.reshape(-1,1))
    funval = vt_min_function_norm_external(data,external_data, x[1:], mask, base_res.Win, base_res.A, \
        num_tests = num_tests,  num_nodes = num_nodes, pred_length = pred_length, train_length = train_length,\
        external_output = external_output, evenspace = evenspace, returnall = returnall)
    return funval
def min_func_wtruth(x, mask, base_data, f_s, true_external_data,\
    base_res, num_tests, num_nodes, pred_length, train_length, scale = True, 
    external_output = True, evenspace = False, returnall = False):
    init_delay = 0
    data = base_data[init_delay:]
    external_data = true_external_data[init_delay:]
    if scale:
        SS = StandardScaler()
        external_data = SS.fit_transform(external_data.reshape(-1,1))
    funval = vt_min_function_norm_external(data,external_data, x, mask, base_res.Win, base_res.A, \
        num_tests = num_tests,  num_nodes = num_nodes, pred_length = pred_length, train_length = train_length,\
        external_output = external_output, evenspace = evenspace, returnall = returnall)
    return funval

num_iters = 5000
num_tests = 20
results_data = np.zeros((num_iters, num_tests))
for i in range(num_iters):
    get_data = True
    data_length = 100000
    step = 0.05
    f_s = 1/step
    scale = 0.01
    slow_var = 48/28
    r_t = lambda x: r_t_const(x)
    dx_dt = lambda x, time, r_t: dxdt_lorenz_rossler(x, time, r_t, scale = scale, slow_var = slow_var)
    if get_data:
        lorenz_data_rossler = getCoupledLorenzData(data_length, r_t, dx_dt, sample_tau = step, seed = i)
        # np.savetxt('lorenz_data_rossler_step%0.2f_scale%0.2f.csv' %(step, scale), lorenz_data_rossler, delimiter = ',')
    else:
        lorenz_data_rossler = np.loadtxt('lorenz_data_rossler_step%0.2f_scale%0.2f.csv' %(step, scale), delimiter = ',')
    scaled_data = lorenz_data_rossler[:,:3]
    scaled_data = np.ascontiguousarray(scaled_data)
    num_nodes = 360
    train_length = 3000
    sync_length = 200
    pred_length = 500
    res_seed = 1
    base_res = reservoir(3,num_nodes,input_weight = 1, spectral_radius = 1, seed = res_seed) #Generate a reservoir
    mask = ['input_weight', 'regularization', 'leakage', 'forget']
    x0 = np.array([4.794673100268221, 6.190777628515722, 2.848775833743345, 8.958038046257439])
    min_func_base = lambda x: vt_min_function_norm(scaled_data, x, mask,\
        base_res.Win, base_res.A, num_nodes, num_tests, sync_length, train_length, pred_length, evenspace = True, returnall = True)
    results = min_func_base(x0)
    results_data[i] = results
    print(i)
    np.savetxt('/lustre/awikner1/Reservoir-GN-RL/rossler_noexternal_results_forget%f.csv' % x0[-1], results_data, delimiter = ',')