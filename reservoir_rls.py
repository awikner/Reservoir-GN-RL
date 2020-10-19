import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg, diags
from numpy.linalg import solve
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numba import jit, float64
from tqdm import tqdm

class reservoir:
    def __init__(self, input_size, approx_num_nodes, input_weight = 1, \
                spectral_radius = 0.9, leakage = 0, average_degree = 3, \
                forget = 0.99, LM_regularization = 1e-10, \
                regularization = 1e-5, delta = 1e6, timestep = 0.05, \
                LM_t_regularization = 1e-6, t_regularization = 1e-6, \
                t_weighted = False, seed = 1):
        # Initializes the reservoir
        self.input_size          = input_size #Number of inputs
        self.num_nodes           = round(np.floor(approx_num_nodes/input_size)*input_size) #Number of nodes
        self.leakage             = leakage
        self.forget              = forget # Exponential forgetting parameter for RLS
        self.LM_regularization   = LM_regularization # Per step regularization in RLS
        self.LM_t_regularization = LM_t_regularization
        self.t_weighted          = t_weighted
        self.regularization      = regularization # Tikhonov Regularization
        self.t_regularization    = t_regularization
        self.delta               = delta # Initial covariance matrix value in RLS
        self.timestep            = timestep
        self.Wout                = np.zeros((input_size, self.num_nodes))
        self.r                   = np.zeros(self.num_nodes)
        self.initializeWin(input_weight, seed) # Initialize the input matrix
        self.initializeA(average_degree, spectral_radius, seed) # Initialize the adjacency matrix
        
    def initializeWin(self, input_weight, seed):
        # Initialize the input matrix. Here, we choose this matrix to be
        # sparse such that each node receives input from only one input.
        np.random.seed(seed) # Set random seed
        self.Win = np.zeros((self.num_nodes, self.input_size))
        nodes_per_input = int(self.num_nodes/self.input_size)
        q     = 0
        qnext = np.copy(nodes_per_input)
        # Assign equal number non-zero input weights to each input
        for i in range(self.input_size):
            self.Win[q:qnext,i] = (-1 + 2*np.random.randn(nodes_per_input))*input_weight
            q = np.copy(qnext)
            qnext += nodes_per_input
            
    def initializeA(self, average_degree, spectral_radius, seed):
        # Initialize the adjacency matrix such that it has a maximum magnitude eigenvalue
        # equal to the spectral radius
        np.random.seed(seed) # Set the random seed
        density = average_degree/self.num_nodes # Set the density
        self.A  = sparse.random(self.num_nodes, self.num_nodes, density = density)
        e       = linalg.eigs(self.A, k = 1, return_eigenvectors=False)
        self.A  = self.A*spectral_radius/np.abs(e[0]) # Rescale to spectral radius       
        
    def advance(self, input):
        # Update the reservoir state using the tanh activation function
        self.r = updateRes(self.r, self.leakage, self.A, self.Win, input)
        
    def trainWout(self, train_data, sync_length):
        # Train the output matrix using the standard ridge regression algorithm
        self.r = np.zeros(self.num_nodes)
        # First, synchronize reservoir to the trajectory
        for t in range(sync_length-1):
            self.advance(train_data[t])
        aug_states = np.zeros((train_data.shape[0]-sync_length,self.num_nodes))
        # Then, begin training over all remaining data points
        for t in range(sync_length, train_data.shape[0]):
            self.advance(train_data[t-1])
            aug_r               = np.copy(self.r)
            aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
            aug_states[t-sync_length] = aug_r
        # Compute Wout using ridge regression
        if not self.t_weighted:
            self.Wout = computeWout(aug_states, train_data[sync_length:], self.forget, self.regularization)
        else:
            self.Wout = computeWoutTweighted(aug_states, train_data[sync_length:], self.forget, \
                self.regularization, self.t_regularization, self.timestep)
        # Advance reservoir state to beginning of test data
        self.advance(train_data[-1])
        
    def trainWoutRLS(self, train_data, sync_length):
        # Train the output matrix using Regularized recursive least squares with forgetting
        self.r = np.zeros(self.num_nodes)
        # Synchronize reservoir state to the trajectory
        for t in range(sync_length-1):
            self.advance(train_data[t])
        # Initialize target and covariance matrices
        data_trstates = np.zeros((self.input_size,self.num_nodes))
        states_trstates_inv = self.delta*np.identity(self.num_nodes)
        inv_forget = 1/self.forget
        for t in range(sync_length, train_data.shape[0]):
            self.advance(train_data[t-1]) #Advance reservoir state
            aug_r               = np.copy(self.r)
            aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
            # Use RRLSwF to update target and covariance matrices
            data_trstates       = updateTarget(data_trstates, aug_r, train_data[t], self.forget) 
            states_trstates_inv = updateCovReg(states_trstates_inv, aug_r, inv_forget, self.LM_regularization)
        # Compute Wout
        self.Wout = data_trstates @ states_trstates_inv
        # Advance reservoir state to beginning of test data
        self.advance(train_data[-1])
        
    def predict(self):
        # Square every other node and apply Wout to obtain a prediction
        aug_r = np.copy(self.r)
        aug_r[::2] = np.power(aug_r[::2],2)
        return self.Wout @ aug_r
        
    def valid_time(self, validation_data, error_bound = 0.2, plot = True):
        # Given an error bound, compute the valid time of prediction.
        # Plot the prediction if specified.
        valid_time = 0
        predictions = np.zeros(validation_data.shape)
        for i in range(validation_data.shape[0]):
            predictions[i] = self.predict() # Obtain prediction
            # Compute error
            error = np.sqrt(np.mean((validation_data[i]-predictions[i])**2))
            if error > error_bound: # If error exceeds threshold, break
                break
            valid_time += 1 # Otherwise, update valid time and feed back prediction
            self.advance(predictions[i])
            
        # Plot prediction if specified
        if plot:
            plot_predictions = predictions[:i]
            plot_validation = validation_data[:i]
            plt.plot(plot_predictions[:,0],label = 'Prediction')
            plt.plot(plot_validation[:,0],label = 'Truth')
            plt.xlabel('Iteration')
            plt.ylabel('x')
            plt.legend()
            plt.show()
            
        return valid_time
            

def cross_validation_performance(data, reservoir, num_tests, sync_length, train_length, pred_length, seed = 10, errormax = 3.2, train_method = 'RLS'):
    # Evaluates the valid time of prediction over a random set of training time series and test time series
    data_length = data.shape[0]
    max_start = data_length - train_length - sync_length - pred_length
    np.random.seed(seed)
    # Obtain starting points for training and predction
    split_starts = np.random.choice(max_start,size = num_tests,replace = False)
    valid_times = np.zeros(num_tests)
    # For each starting point
    with tqdm(total = num_tests) as pbar:
        for i in range(num_tests):
            train_data      = data[split_starts[i]:split_starts[i]+sync_length+train_length] # Get training data
            # Get validation data
            validation_data = data[split_starts[i]+sync_length+train_length:split_starts[i]+sync_length+train_length+pred_length]
            # Train the reservoir
            if train_method is 'RLS':
                reservoir.trainWoutRLS(train_data, sync_length)
            else:
                reservoir.trainWout(train_data, sync_length)
            # Evaluate the prediction
            valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
            pbar.update(1)
        
    return valid_times

def cross_validation_performance_wscaling(data, reservoir, num_tests, sync_length, train_length, pred_length, seed = 10, errormax = 0.2):
    data_length = data.shape[0]
    max_start = data_length - train_length - sync_length - pred_length
    np.random.seed(seed)
    split_starts = np.random.choice(max_start,size = num_tests,replace = False)
    valid_times = np.zeros(num_tests)
    for i in range(num_tests):
        train_data      = data[split_starts[i]:split_starts[i]+sync_length+train_length]
        scaler = preprocessing.StandardScaler().fit(train_data)
        train_data_scaled = scaler.transform(train_data)
        validation_data_scaled = scaler.transform(data[split_starts[i]+sync_length+train_length:\
            split_starts[i]+sync_length+train_length+pred_length])
        reservoir.trainWoutRLS(train_data, sync_length)
        valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
        
    return valid_times

# @jit(nopython = True, fastmath = True)
def updateTarget(data_trstates, aug_r, input, forget):
    data_trstates = forget*data_trstates + np.outer(input,aug_r)
    return data_trstates

# @jit(nopython = True, fastmath = True)
def updateCovReg(states_trstates_inv, aug_r, inv_forget, LM_regularization):
    states_trstates_inv = inv_forget*(states_trstates_inv - 1.0/\
        (1+inv_forget*np.dot(np.dot(aug_r,states_trstates_inv),aug_r))*\
        np.outer(np.dot(states_trstates_inv,aug_r),np.dot(aug_r,states_trstates_inv))\
        *inv_forget)
    states_trstates_inv = states_trstates_inv - \
        LM_regularization*(states_trstates_inv @ states_trstates_inv)
    return states_trstates_inv

# @jit(nopython = True, fastmath = True)
def updateRes(r, leakage, A, Win, input):
    newr = leakage*r + (1-leakage)*\
            np.tanh(A.dot(r) + np.dot(Win,input))
    return newr

# @jit(nopython = True, fastmath = True)
def computeWout(states, truth, forget, regularization):
    n = states.shape[0]
    d = states.shape[1]
    s_mat = np.diag(np.power(forget,np.arange(n)[::-1]))
    data_trstates = truth.T @ s_mat @ states
    states_trstates = states.T @ s_mat @ states + regularization * sparse.identity(d)
    WoutT = solve(states_trstates.T, data_trstates.T)
    return WoutT.T

def computeWoutTweighted(states, truth, forget, regularization,t_regularization, step):
    n = states.shape[0]
    d = states.shape[1]
    s_mat = np.diag(np.power(forget,np.arange(n)[::-1]))
    t_scale_mat = sparse.diags(np.arange((1-n)*step,step,step))
    reg_mat = np.zeros((2*d,2*d))
    reg_mat[:d,:d] = np.diag(regularization*np.ones(d))
    reg_mat[d:,d:] = np.diag(t_regularization*np.ones(d))
    aug_states = np.append(states, t_scale_mat @ states, axis = 1)
    data_trstates = truth.T @ s_mat @ aug_states
    states_trstates = aug_states.T @ s_mat @ aug_states + reg_mat
    WoutT = solve(states_trstates.T, data_trstates.T)
    return WoutT.T