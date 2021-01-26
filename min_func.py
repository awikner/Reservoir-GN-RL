import numpy as np
import cma
from reservoir_rls import *
from lorenz63 import *

def min_func(seed):
    data_length = 100000
    step = 0.05
    r_t = lambda x: r_t_const(x)
    dxdt = lambda x,t,r_t: dxdt_lorenz(x,t,r_t)
    data = getLorenzData(data_length, r_t, dxdt, sample_tau = step)

    sync_length = 200
    num_tests = 500
    train_length = 400
    pred_length = 400

    # scaler = preprocessing.StandardScaler().fit(data)
    # scaled_data = scaler.transform(data)
    scaled_data = np.copy(data)
    scaled_data = np.ascontiguousarray(scaled_data)
    
    mask = [0, 2]
    min_func = lambda x: vt_min_function_norm(scaled_data, x, mask, num_nodes = 210, pred_length = 500, seed = seed)
    x0 = np.array([0, 5])
    sigma = 0.5
    
    opts = cma.CMAOptions()
    opts.set('popsize',10*x0.size)
    opts.set('bounds', [0,10])
    results = cma.fmin(min_func, x0, sigma, options = opts)
    return results