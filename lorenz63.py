import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

def ornstein_uhlenbeck_em(data_length,transient_length = 1000, \
        tau = 0.01,sample_tau = 0.05, seed = 5, theta = 1, sigma = 1):
    sampling_rate = round(sample_tau/tau)
    np.random.seed(seed)
    x = np.random.randn(1)
    for i in range(transient_length):
        x += euler_maruyama(-theta*x, sigma, tau)
    data = np.zeros(data_length)
    data[0] = x
    for i in range(1,data_length):
        data[i] = data[i-1] + euler_maruyama(-theta*data[i-1], sigma, tau)
    data = data[::sampling_rate]
    return data

def euler_maruyama(a,b,tau):
    return a*tau + b*np.sqrt(tau)*np.random.randn(1)
    
def dxdt_lorenz(x,time,r_t, sigma = 10., beta = 8/3, rho = 28.):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    return np.array([sigma*(- x[0] + x[1]),\
                     r_t(time)*rho*x[0] - x[1] - x[0]*x[2],\
                     x[0]*x[1]-beta*x[2]])

def dxdt_lorenz_scaled(x,time,r_t, sigma = 10., beta = 8/3, rho = 28., scale = 1., S = 1.):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    return np.array([sigma*(- x[0] + x[1]),\
                     r_t(time)*rho*x[0] - x[1] - S*x[0]*x[2],\
                     S*x[0]*x[1]-beta*x[2]])*scale

def dxdt_lorenz_2scale(x,time,r_t, sigma = 10, beta = 8/3, rho = 28, C1 = .15, \
                       C2 = .15, C1star = 0, C2star = 0, S = 0.1, O = 0, scale = 0.1):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    dx = sigma*(-x[0]+x[1])-C1*(S*x[3]+O)
    dy = r_t(time)*rho*x[0]-x[1]-x[0]*x[2]+C1*(S*x[4]+O)
    dz = x[0]*x[1]-beta*x[2]+C1star*x[5]
    
    dX = (sigma*(-x[3]+x[4]))*scale - C2*(x[0]+O)
    dY = (r_t(time)*rho*x[3]-x[4]-S*x[3]*x[5])*scale + C2*(x[1]+O)
    dZ = (S*x[3]*x[4]-beta*x[5])*scale-C2star*x[2]
               
    return np.array([dx,dy,dz,dX,dY,dZ])

def dxdt_rossler(x,time,r_t, a = 0.2, b = 0.2, c = 5.7, scale = 0.1):
    dX = (-x[1]-x[2])*scale
    dY = (x[0]+a*x[1])*scale
    dZ = (b + x[2]*(x[0]-c))*scale
    return np.array([dX,dY,dZ])

def dxdt_lorenz_rossler(x,time,r_t, sigma = 10, beta = 8/3, rho = 28, a = 0.2, b = 0.2, c = 5.7, \
                        slow_var = 48/28, slow_max = 7.84, slow_min = -10.80, scale = 1):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    rossler_scaling = (x[4]-(slow_max+slow_min)/2)/(slow_max-slow_min)*(slow_var-1)+(1+(slow_var-1)/2)
    dx = sigma*(-x[0]+x[1])
    dy = r_t(time)*rho*rossler_scaling*x[0]-x[1]-x[0]*x[2]
    dz = x[0]*x[1]-beta*x[2]
    
    dX = (-x[4]-x[5])*scale
    dY = (x[3]+a*x[4])*scale
    dZ = (b + x[5]*(x[3]-c))*scale
               
    return np.array([dx,dy,dz,dX,dY,dZ])

def dxdt_lorenz_coupled(x,time,r_t, L1 = 8/3,L2 = 8/3, R1 = 70, R2 = 28, P1 = 10, P2 = 10, mu = 0.1):
    # Evaluates derivative of Lorenz '63 system with a time-dependent
    # rho value. For constant rho, input the r_t_const function.
    da = 1 - x[0] - x[1]*x[2]
    db = x[0]*x[2]-x[1]-L1*(x[1]-x[4]/mu)
    dc = P1*(R1*mu*x[1]-x[2])
    
    dA = 1-x[3]-x[4]*x[5]
    dB = x[5]*x[3]-x[4]-L2*(x[4]-mu*x[1])
    dC = P2*(R2*x[4]-x[5])
               
    return np.array([da,db,dc,dA,dB,dC])
    
def rk4(x, time, tau, r_t, dxdt):
    # Fourth order Runge-Kutta integrator
    
    k1 = dxdt(x, time, r_t)
    k2 = dxdt(x + k1/2*tau, time + tau/2, r_t)
    k3 = dxdt(x + k2/2*tau, time + tau/2, r_t)
    k4 = dxdt(x + tau*k3, time + tau, r_t)
    
    xnext = x + 1/6*tau*(k1+2*k2+2*k3+k4)
    return xnext

def getLorenzData(data_length, r_t, dxdt_lorenz,transient_length = 1000, tau = 0.01,sample_tau = 0.05, seed = 5):
    # Obtains time series of Lorenz '63 states after some initial transient time
    sampling_rate = round(sample_tau/tau)
    np.random.seed(seed)
    x = np.random.rand(3)
    time = -transient_length*tau
    for i in range(0,transient_length):
        x = rk4(x,time,tau,r_t,dxdt_lorenz)
        time += tau
    
    data = np.zeros((3,data_length))
    data[:,0] = x
    for i in range(0,data_length-1):
        data[:,i+1] = rk4(data[:,i],time,tau,r_t,dxdt_lorenz)
        time += tau
    
    # print(np.mean(data[0,:]))
    # print(np.max(data[0,:]))
    # print(np.min(data[0,:]))
    data = data[:,::sampling_rate].T
    return data

def getRosslerData(data_length, r_t, dxdt_lorenz,transient_length = 1000, tau = 0.01,sample_tau = 0.05, seed = 5):
    # Obtains time series of Lorenz '63 states after some initial transient time
    sampling_rate = round(sample_tau/tau)
    np.random.seed(seed)
    x = np.random.rand(3)-np.array([0,5,0])
    time = -transient_length*tau
    for i in range(0,transient_length):
        x = rk4(x,time,tau,r_t,dxdt_lorenz)
        time += tau
    
    data = np.zeros((3,data_length))
    data[:,0] = x
    for i in range(0,data_length-1):
        data[:,i+1] = rk4(data[:,i],time,tau,r_t,dxdt_lorenz)
        time += tau
    
    # print(np.mean(data[0,:]))
    # print(np.max(data[0,:]))
    # print(np.min(data[0,:]))
    data = data[:,::sampling_rate].T
    return data

def getCoupledLorenzData(data_length, r_t, dxdt_lorenz, transient_length = 1000, tau = 0.01,sample_tau = 0.05, seed = 5):
    # Obtains time series of Lorenz '63 states after some initial transient time
    sampling_rate = round(sample_tau/tau)
    np.random.seed(seed)
    x = np.random.rand(6)+np.array([0,0,0,0,-5,0])
    time = -transient_length*tau
    for i in range(0,transient_length):
        x = rk4(x,time,tau,r_t,dxdt_lorenz)
        time += tau
    
    data = np.zeros((6,data_length))
    data[:,0] = x
    for i in range(0,data_length-1):
        data[:,i+1] = rk4(data[:,i],time,tau,r_t,dxdt_lorenz)
        time += tau
    
    data = data[:,::sampling_rate].T
    return data

def get2ScaleLorenzData(data_length, r_t, dxdt_lorenz, transient_length = 1000, tau = 0.01,sample_tau = 0.05, seed = 5):
    # Obtains time series of Lorenz '63 states after some initial transient time
    sampling_rate = round(sample_tau/tau)
    np.random.seed(seed)
    x = np.random.rand(6)
    real_transient_length = int(transient_length/tau)
    time = -real_transient_length*tau
    for i in range(0,real_transient_length):
        x = rk4(x,time,tau,r_t,dxdt_lorenz)
        time += tau
    
    data = np.zeros((6,data_length))
    data[:,0] = x
    for i in range(0,data_length-1):
        data[:,i+1] = rk4(data[:,i],time,tau,r_t,dxdt_lorenz)
        time += tau
    
    data = data[:,::sampling_rate].T
    return data

def r_t_cosine(time, period = 500, max_height = 48/28):
    # Function for oscillating rho value (not used here)
    r = 1 + (max_height-1.)/2 - (max_height-1)/2*np.cos(2*np.pi/period*time)
    return r

def r_t_const(time, value = 1):
    # Function for constant rho value
    return value

def r_t_extern(time, external_r, tau = 0.01):
    idx = round(time/tau)
    return external_r[idx]

def separate_lorenz_2scale(data, f_s, Wn_xy, Wn_z, filter_order = 10):
    sos_xy = butter(filter_order, Wn_xy, 'lowpass', fs = f_s, output = 'sos')
    sos_z  = butter(filter_order, Wn_z, 'lowpass', fs = f_s, output = 'sos')
    filtered_xy = sosfilt(sos_xy, data[:,:2], axis = 0)
    filtered_z  = sosfilt(sos_z, data[:,2])
    slow_data   = np.hstack((filtered_xy, filtered_z.reshape(-1,1)))
    fast_data   = data-slow_data
    return np.append(fast_data, slow_data,axis = 1)
    
    