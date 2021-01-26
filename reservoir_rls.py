import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg, diags, csr_matrix
from scipy.special import expit
from numpy.linalg import solve
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numba import jit, float64
from tqdm import tqdm
import time
# from RLS_functions_2 import updateRes, updateCovReg, updateTarget, updateCovReg_tweighted, updateTarget_tweighted

class reservoir:
    def __init__(self, input_size, approx_num_nodes, input_weight = 1, \
                spectral_radius = 0.9, leakage = 0, average_degree = 3, \
                forget = 0.99, LM_regularization = 1e-10, \
                regularization = 1e-6, delta = 1e6, timestep = 0.05, \
                LM_t_regularization = 1e-6, t_regularization = 1e-6, \
                t_weighted = False, RLS_reg_type = 'LM', Win=np.nan, A = np.nan, seed = 1):
        # Initializes the reservoir
        self.input_size          = input_size #Number of inputs
        self.num_nodes           = int(round(np.floor(approx_num_nodes/input_size)*input_size)) #Number of nodes
        self.leakage             = leakage
        self.forget              = forget # Exponential forgetting parameter for RLS
        self.LM_regularization   = LM_regularization # Per step regularization in RLS
        self.LM_t_regularization = LM_t_regularization
        self.t_weighted          = t_weighted
        self.RLS_reg_type        = RLS_reg_type
        self.regularization      = regularization # Tikhonov Regularization
        self.t_regularization    = t_regularization
        self.delta               = delta # Initial covariance matrix value in RLS
        self.timestep            = timestep
        self.Wout                = np.zeros((input_size, self.num_nodes))
        self.r                   = np.zeros(self.num_nodes)
        if np.all(np.isnan(Win)):
            self.initializeWin(input_weight, seed) # Initialize the input matrix
        else:
            self.Win = Win*input_weight
        if type(A) is not sparse.coo.coo_matrix:
            self.initializeA(average_degree, spectral_radius, seed) # Initialize the adjacency matrix
        else:
            self.A = A*spectral_radius
        
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
        # plt.hist(aug_states[:,1::2].flatten(), 50)
        # plt.show()
        # plt.hist(aug_states[:,::2].flatten(), 50)
        # plt.show()
        if not self.t_weighted:
            self.Wout, self.aug_states, self.target_data = computeWout(aug_states, train_data[sync_length:], self.forget, self.regularization)
        else:
            self.Wout, self.aug_states, self.target_data = computeWoutTweighted(aug_states, train_data[sync_length:], self.forget, \
                self.regularization, self.t_regularization, self.timestep)
        plt.plot(np.linalg.norm((np.transpose(self.Wout @ aug_states.T) - train_data[sync_length:])/np.std(train_data[sync_length:], axis = 0), axis = 1))
        plt.show()
        # Advance reservoir state to beginning of test data
        self.advance(train_data[-1])
        
    def updateWout(self, new_train_data):
        new_aug_states = np.zeros((new_train_data.shape[0], self.num_nodes))
        aug_r = np.copy(self.r)
        aug_r[::2] = np.power(aug_r[::2],2)
        new_aug_states[0] = aug_r
        for t in range(new_train_data.shape[0]-1):
            self.advance(new_train_data[t])
            aug_r = np.copy(self.r)
            aug_r[::2] = np.power(aug_r[::2],2)
            new_aug_states[t+1] = aug_r
        
        aug_states = np.vstack((self.aug_states, new_aug_states))
        target_data = np.vstack((self.target_data, new_train_data))
        if not self.t_weighted:
            self.Wout, self.aug_states, self.target_data = computeWout(aug_states, target_data, self.forget, self.regularization)
        else:
            self.Wout, self.aug_states, self.target_data = computeWoutTweighted(aug_states, target_data, self.forget, \
                self.regularization, self.t_regularization, self.timestep)
        self.advance(new_train_data[-1])
        
    def trainWoutRLS(self, train_data, sync_length):
        # Train the output matrix using Regularized recursive least squares with forgetting
        self.r = np.zeros(self.num_nodes)
        # Synchronize reservoir state to the trajectory
        for t in range(sync_length-1):
            self.advance(train_data[t])
        inv_forget = 1/self.forget
        # Initialize target and covariance matrices
        if not self.t_weighted:
            self.data_trstates = np.zeros((self.input_size,self.num_nodes))
            self.states_trstates_inv = self.delta*np.identity(self.num_nodes)
            for t in range(sync_length, train_data.shape[0]):
                self.advance(train_data[t-1]) #Advance reservoir state
                aug_r               = np.copy(self.r)
                aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
                # Use RRLSwF to update target and covariance matrices
                regularization_mat       = self.LM_regularization * np.ones(self.num_nodes).reshape(-1,1)
                # tic = time.perf_counter()
                self.data_trstates       = updateTarget(self.data_trstates, aug_r, train_data[t], self.forget)
                # toc = time.perf_counter()
                # print(toc - tic)
                # tic = time.perf_counter()
                self.states_trstates_inv = updateCovReg(self.states_trstates_inv, aug_r, inv_forget, regularization_mat, self.RLS_reg_type)
                # toc = time.perf_counter()
                # print(toc - tic)
            # Compute Wout
            if self.RLS_reg_type == 'LM':
                self.Wout = self.data_trstates @ self.states_trstates_inv
            else:
                self.Wout = self.data_trstates @ solve(np.identity(self.states_trstates_inv.shape[0]) + \
                    self.regularization * self.states_trstates_inv, self.states_trstates_inv)
        else:
            self.data_trstates = np.zeros((self.input_size,2*self.num_nodes))
            self.states_trstates_inv = self.delta*np.identity(2*self.num_nodes)
            t_update_mat_T = csr_matrix(np.append(np.append(np.identity(self.num_nodes),np.zeros((self.num_nodes,self.num_nodes)),axis = 1),\
                 np.append(-self.timestep*np.identity(self.num_nodes),np.identity(self.num_nodes),axis = 1), axis = 0).T)
            t_update_mat_inv = csr_matrix(np.append(np.append(np.identity(self.num_nodes),np.zeros((self.num_nodes,self.num_nodes)),axis = 1),\
                 np.append(self.timestep*np.identity(self.num_nodes),np.identity(self.num_nodes),axis = 1), axis = 0))
            regularization_mat = np.append(np.ones(self.num_nodes)*self.LM_regularization,\
                                           np.ones(self.num_nodes)*self.LM_t_regularization).reshape(-1,1)
            for t in range(sync_length, train_data.shape[0]):
                self.advance(train_data[t-1]) #Advance reservoir state
                aug_r               = np.copy(self.r)
                aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
                aug_r_tweighted     = np.append(aug_r,np.zeros(self.num_nodes))
                # Use RRLSwF to update target and covariance matrices
                self.data_trstates       = updateTarget_tweighted(self.data_trstates, aug_r_tweighted, train_data[t], self.forget, t_update_mat_T) 
                self.states_trstates_inv = updateCovReg_tweighted(self.states_trstates_inv, aug_r_tweighted, inv_forget,t_update_mat_inv,\
                    regularization_mat, self.RLS_reg_type)
            # Compute Wout
            if self.RLS_reg_type == 'LM':
                self.Wout = self.data_trstates @ self.states_trstates_inv
            else:
                regularization_mat = np.append(np.ones(self.num_nodes)*self.regularization,\
                    np.ones(self.num_nodes)*self.t_regularization).reshape(-1,1)
                reg_states_trstates_inv = solve(np.identity(self.states_trstates_inv.shape[0]) + \
                    regularization_mat * self.states_trstates_inv.T, self.states_trstates_inv.T).T
                self.Wout = self.data_trstates @ reg_states_trstates_inv
        
        # Advance reservoir state to beginning of test data
        self.advance(train_data[-1])
        
    def updateWoutRLS(self, new_train_data):
        # Train the output matrix using Regularized recursive least squares with forgetting
        inv_forget = 1/self.forget
        # Initialize target and covariance matrices
        if not self.t_weighted:
            aug_r               = np.copy(self.r)
            aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
            # Use RRLSwF to update target and covariance matrices
            regularization_mat       = self.LM_regularization * np.ones(self.num_nodes).reshape(-1,1)
            self.data_trstates       = updateTarget(self.data_trstates, aug_r, new_train_data[0], self.forget) 
            self.states_trstates_inv = updateCovReg(self.states_trstates_inv, aug_r, inv_forget, regularization_mat, self.RLS_reg_type)
            for t in range(new_train_data.shape[0]-1):
                self.advance(new_train_data[t]) #Advance reservoir state
                aug_r               = np.copy(self.r)
                aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
                # Use RRLSwF to update target and covariance matrices
                self.data_trstates       = updateTarget(self.data_trstates, aug_r, new_train_data[t+1], self.forget) 
                self.states_trstates_inv = updateCovReg(self.states_trstates_inv, aug_r, inv_forget, regularization_mat, self.RLS_reg_type)
            # Compute Wout
            if self.RLS_reg_type == 'LM':
                self.Wout = self.data_trstates @ self.states_trstates_inv
            else:
                self.Wout = self.data_trstates @ solve(np.identity(self.states_trstates_inv.shape[0]) + \
                   self.regularization * self.states_trstates_inv, self.states_trstates_inv)
        else:
            t_update_mat_T = csr_matrix(np.append(np.append(np.identity(self.num_nodes),np.zeros((self.num_nodes,self.num_nodes)),axis = 1),\
                 np.append(-self.timestep*np.identity(self.num_nodes),np.identity(self.num_nodes),axis = 1), axis = 0).T)
            t_update_mat_inv = csr_matrix(np.append(np.append(np.identity(self.num_nodes),np.zeros((self.num_nodes,self.num_nodes)),axis = 1),\
                 np.append(self.timestep*np.identity(self.num_nodes),np.identity(self.num_nodes),axis = 1), axis = 0))
            regularization_mat = np.append(np.ones(self.num_nodes)*self.LM_regularization,\
                                           np.ones(self.num_nodes)*self.LM_t_regularization).reshape(-1,1)
            aug_r               = np.copy(self.r)
            aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
            aug_r_tweighted     = np.append(aug_r,np.zeros(self.num_nodes))
            # Use RRLSwF to update target and covariance matrices
            self.data_trstates       = updateTarget_tweighted(self.data_trstates, aug_r_tweighted, new_train_data[0], self.forget, t_update_mat_T) 
            self.states_trstates_inv = updateCovReg_tweighted(self.states_trstates_inv, aug_r_tweighted, inv_forget,t_update_mat_inv,\
                regularization_mat, self.RLS_reg_type)
            for t in range(new_train_data.shape[0]-1):
                self.advance(new_train_data[t]) #Advance reservoir state
                aug_r               = np.copy(self.r)
                aug_r[::2]          = np.power(aug_r[::2],2) # Square every other node
                aug_r_tweighted     = np.append(aug_r,np.zeros(self.num_nodes))
                # Use RRLSwF to update target and covariance matrices
                self.data_trstates       = updateTarget_tweighted(self.data_trstates, aug_r_tweighted, \
                    new_train_data[t+1], self.forget,t_update_mat_T) 
                self.states_trstates_inv = updateCovReg_tweighted(self.states_trstates_inv, aug_r_tweighted, inv_forget,t_update_mat_inv,\
                    regularization_mat, self.RLS_reg_type)
            # Compute Wout
            if self.RLS_reg_type == 'LM':
                self.Wout = self.data_trstates @ self.states_trstates_inv
            else:
                regularization_mat = np.append(np.ones(self.num_nodes)*self.regularization,\
                    np.ones(self.num_nodes)*self.t_regularization).reshape(-1,1)
                reg_states_trstates_inv = solve(np.identity(self.states_trstates_inv.shape[0]) + \
                    regularization_mat * self.states_trstates_inv.T, self.states_trstates_inv.T).T
                self.Wout = self.data_trstates @ reg_states_trstates_inv
        # Advance reservoir state to beginning of test data
        self.advance(new_train_data[-1])
        
    def synchronize(self, sync_data):
        sync_length = sync_data.shape[0]
        self.r = np.zeros(self.num_nodes)
        # Synchronize reservoir state to the trajectory
        for t in range(sync_length):
            self.advance(sync_data[t])
                
    def predict(self):
        # Square every other node and apply Wout to obtain a prediction
        aug_r = np.copy(self.r)
        aug_r[::2] = np.power(aug_r[::2],2)
        return self.Wout @ aug_r
    
    def predict_tweighted(self, time):
        # Square every other node and apply Wout to obtain a prediction
        aug_r = np.copy(self.r)
        aug_r[::2] = np.power(aug_r[::2],2)
        aug_r_tweighted = np.append(aug_r, time * aug_r)
        return self.Wout @ aug_r_tweighted
        
    def valid_time(self, validation_data, error_bound = 0.2, plot = False):
        # Given an error bound, compute the valid time of prediction.
        # Plot the prediction if specified.
        valid_time = 0
        predictions = np.zeros(validation_data.shape)
        errors = np.zeros(validation_data.shape[0])
        initial_r = np.copy(self.r)
        if not self.t_weighted:
            for i in range(validation_data.shape[0]):
                predictions[i] = self.predict() # Obtain prediction
                # Compute error
                error = np.sqrt(np.mean((validation_data[i]-predictions[i])**2))
                errors[i] = error
                if error > error_bound: # If error exceeds threshold, break
                    break
                valid_time += 1 # Otherwise, update valid time and feed back prediction
                self.advance(predictions[i])
        else:
            time = np.copy(self.timestep)
            for i in range(validation_data.shape[0]):
                predictions[i] = self.predict_tweighted(time) # Obtain prediction
                # Compute error
                error = np.sqrt(np.mean((validation_data[i]-predictions[i])**2))
                errors[i] = error
                if error > error_bound: # If error exceeds threshold, break
                    break
                valid_time += 1 # Otherwise, update valid time and feed back prediction
                self.advance(predictions[i])
                time += self.timestep
            
        # if valid_time == validation_data.shape[0]-1:
        #     raise ValueError
        # Plot prediction if specified
        if plot:
            plt.rcParams.update({'font.size': 18})

            """
            plot_predictions = predictions[:i]
            plot_validation = validation_data[:i]
            plt.plot((np.arange(plot_predictions[:,0].size)+1)*self.timestep, plot_predictions[:,0],label = 'Prediction')
            plt.plot((np.arange(plot_predictions[:,0].size)+1)*self.timestep, plot_validation[:,0],label = 'Truth')
            plt.xlabel('Time')
            plt.ylabel('x')
            plt.legend()
            # plt.savefig('prediction_vt%d.svg' % valid_time)
            plt.show()
            """
            max_plot_time = int(8/0.05)
            plot_predictions = predictions[:i]
            plot_validation = validation_data[:i]
            
            fig1 = plt.figure(figsize = (7,2))
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, plot_predictions[:max_plot_time,0])
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, plot_validation[:max_plot_time,0])
            # plt.xlabel('Time')
            plt.ylabel('x(t)')
            plt.xlim(0,8)
            plt.ylim(-25,25)
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            plt.savefig('prediction_x_forget%0.2f.svg' % self.forget)
            plt.show()
            
            fig2 = plt.figure(figsize = (7,2))
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, plot_predictions[:max_plot_time,1])
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, plot_validation[:max_plot_time,1])
            # plt.xlabel('Time')
            plt.ylabel('y(t)')
            plt.xlim(0,8)
            plt.ylim(-30,30)
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            plt.savefig('prediction_y_forget%0.2f.svg' % self.forget)
            plt.show()
            
            fig3 = plt.figure(figsize = (7,2))
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, plot_predictions[:max_plot_time,2])
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, plot_validation[:max_plot_time,2])
            # plt.xlabel('Time')
            plt.ylabel('z(t)')
            plt.xlim(0,8)
            plt.ylim(0,60)
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False) # labels along the bottom edge are off
            plt.savefig('prediction_z_forget%0.2f.svg' % self.forget)
            plt.show()
            
            fig4 = plt.figure(figsize = (7,2))
            print(error_bound)
            plt.plot(np.array([0,8]),np.array([3.2,3.2]))
            plt.plot((np.arange(max_plot_time)+1)*self.timestep, errors[:max_plot_time])
            # plt.xlabel('Time')
            plt.ylabel('RMS Error')
            plt.xlabel('Time')
            plt.xlim(0,8)
            plt.ylim(0,30)
            plt.savefig('prediction_rms_forget%0.2f.svg' % self.forget)
            plt.show()
        
        self.r = initial_r
        return valid_time
            
def vt_min_function_norm(data, hyperparams, mask, base_Win, base_A, num_nodes = 210, \
                         num_tests = 200, sync_length = 200, train_length = 400, \
                         pred_length = 400, separated = False):
    # print(hyperparams)
    input_weight = 0.01
    spectral_radius = 0.9
    regularization = 1e-7
    forget  = 1
    leakage = 0
    for i in range(len(mask)):
        if mask[i] == 'input_weight':
            input_weight = 0.2*(hyperparams[i]-5)
        elif mask[i] == 'spectral_radius':
            spectral_radius = hyperparams[i]
        elif mask[i] == 'regularization':
            regularization = 10.**(-hyperparams[i]-3.0)
        elif mask[i] == 'leakage':
            leakage = expit(hyperparams[i])
        elif mask[i] == 'forget':
            forget = 1 - 10**(-0.5*hyperparams[i]-1.)
        else:
            raise ValueError
        
    res = reservoir(data.shape[1], num_nodes, Win = base_Win, A = base_A, forget = forget, \
        input_weight = input_weight, spectral_radius = spectral_radius, \
        regularization = regularization, leakage = leakage)
    
    if separated:
        valid_times = cross_validation_performance_separated(data, res, num_tests, sync_length, \
                                                   train_length, pred_length, \
                                                   train_method = 'Normal')
    else:
        valid_times = cross_validation_performance_resync(data, res, num_tests, sync_length, \
                                                   train_length, pred_length, \
                                                   train_method = 'Normal')
    # print('Input Weight: %e, Reg: %e' % (input_weight, regularization))
    return -np.median(valid_times)

def vt_min_function_rls(data, hyperparams, mask, base_Win, base_A, num_nodes = 210, \
                         num_tests = 200, sync_length = 200, train_length = 400, \
                         pred_length = 400, sampling = 'random', progress = False):
    input_weight = 0.2*(5.066848935533839-5)
    spectral_radius = 0.9
    LM_regularization = 1e-10
    forget  = 1
    leakage = 0
    param_str = ''
    for i in range(len(mask)):
        if mask[i] == 'input_weight':
            input_weight = 0.2*(hyperparams[i]-5.)
            param_str = param_str + 'Input Weight: %f, ' % (input_weight) 
        elif mask[i] == 'spectral_radius':
            spectral_radius = expit(hyperparams[i])
            param_str = param_str + 'Spectral Radius: %f, ' % (spectral_radius) 
        elif mask[i] == 'LM_regularization':
            LM_regularization = 10.**(-3/5*hyperparams[i]-6.)
            param_str = param_str + 'LM Reg: %e, ' % (LM_regularization) 
        elif mask[i] == 'leakage':
            leakage = expit(hyperparams[i])
            param_str = param_str + 'Leakage: %f, ' % (leakage) 
        elif mask[i] == 'forget':
            forget = 1 - 10**(-0.5*hyperparams[i]-1.)
            param_str = param_str + 'Forget: %f, ' % (forget) 
        else:
            raise ValueError
    tic = time.perf_counter()
    res = reservoir(data.shape[1], num_nodes, Win = base_Win, A = base_A, forget = forget, \
        input_weight = input_weight, spectral_radius = spectral_radius, \
        LM_regularization = LM_regularization, leakage = leakage, RLS_reg_type = 'LM')
    if sampling == 'periodic':
        valid_times = cross_validation_performance_separated(data, res, num_tests, sync_length, \
            train_length, pred_length, train_method = 'RLS', progress = progress)
    elif sampling == 'random':
        valid_times = cross_validation_performance_resync(data, res, num_tests, sync_length, \
            train_length, pred_length, train_method = 'RLS', progress = progress)
    toc = time.perf_counter()
    param_str = param_str + 'Runtime: %f sec.' % (toc - tic)
    # print(param_str)
    return -np.median(valid_times)

def cross_validation_performance(data, reservoir, num_tests, sync_length, train_length, pred_length, \
                                 seed = 5, errormax = 3.2, train_method = 'RLS', progress = False):
    # Evaluates the valid time of prediction over a random set of training time series and test time series
    data_length = data.shape[0]
    max_start = data_length - train_length - sync_length - pred_length
    if not np.isnan(seed):
        np.random.seed(seed)
    # Obtain starting points for training and predction
    split_starts = np.random.choice(max_start,size = num_tests,replace = False)
    valid_times = np.zeros(num_tests)
    # For each starting point
    if progress: 
        with tqdm(total = num_tests) as pbar:
            for i in range(num_tests):
                train_data      = data[split_starts[i]:split_starts[i]+sync_length+train_length] # Get training data
                # Get validation data
                validation_data = data[split_starts[i]+sync_length+train_length:split_starts[i]+sync_length+train_length+pred_length]
                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.trainWoutRLS(train_data, sync_length)
                else:
                    reservoir.trainWout(train_data, sync_length)
                # Evaluate the prediction
                valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
                pbar.update(1)
    else:
        for i in range(num_tests):
            train_data      = data[split_starts[i]:split_starts[i]+sync_length+train_length] # Get training data
            # Get validation data
            validation_data = data[split_starts[i]+sync_length+train_length:split_starts[i]+sync_length+train_length+pred_length]
            # Train the reservoir
            if train_method == 'RLS':
                reservoir.trainWoutRLS(train_data, sync_length)
            else:
                reservoir.trainWout(train_data, sync_length)
            # Evaluate the prediction
            valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
        
    return valid_times

def cross_validation_performance_resync(data, reservoir, num_tests, sync_length, train_length, pred_length, \
                                 seed = 5, errormax = 3.2, train_method = 'RLS', progress = False):
    # Evaluates the valid time of prediction over a random set of training time series and test time series
    data_length = data.shape[0]
    max_train_start = data_length - train_length - sync_length
    max_pred_start  = data_length - pred_length - sync_length
    preds_per_train = 1
    if not np.isnan(seed):
        np.random.seed(seed)
    # Obtain starting points for training and predction
    split_train_starts = np.random.choice(max_train_start,size = num_tests,replace = False)
    split_pred_starts = np.zeros(num_tests)
    for i in range(num_tests):
        if split_train_starts[i] < sync_length + pred_length:
            split_pred_starts[i] = np.random.choice(np.arange(split_train_starts[i] + sync_length + train_length, max_pred_start), \
                                                    size = preds_per_train)
        elif split_train_starts[i] + sync_length + train_length > max_pred_start:
            split_pred_starts[i] = np.random.choice(np.arange(split_train_starts[i] - sync_length - pred_length), size = preds_per_train)
        else:
            split_pred_starts[i] = np.random.choice(np.append(np.arange(split_train_starts[i] - sync_length - pred_length),\
                np.arange(split_train_starts[i] + sync_length + train_length, max_pred_start)), size = preds_per_train)
    split_pred_starts = [int(i) for i in split_pred_starts]
    valid_times = np.zeros(num_tests)
    # For each starting point
    if progress: 
        with tqdm(total = num_tests) as pbar:
            for i in range(num_tests):
                train_data      = data[split_train_starts[i]:split_train_starts[i]+sync_length+train_length] # Get training data
                # Get validation data
                resync_data = data[split_pred_starts[i]:split_pred_starts[i]+sync_length]
                validation_data = data[split_pred_starts[i]+sync_length:split_pred_starts[i]+sync_length+pred_length]
                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.trainWoutRLS(train_data, sync_length)
                else:
                    reservoir.trainWout(train_data, sync_length)
                # Evaluate the prediction
                reservoir.synchronize(resync_data)
                valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
                pbar.update(1)
    else:
        for i in range(num_tests):
            train_data      = data[split_train_starts[i]:split_train_starts[i]+sync_length+train_length] # Get training data
            # Get validation data
            resync_data = data[split_pred_starts[i]:split_pred_starts[i]+sync_length]
            validation_data = data[split_pred_starts[i]+sync_length:split_pred_starts[i]+sync_length+pred_length]
            # Train the reservoir
            if train_method == 'RLS':
                reservoir.trainWoutRLS(train_data, sync_length)
            else:
                reservoir.trainWout(train_data, sync_length)
            # Evaluate the prediction
            reservoir.synchronize(resync_data)
            valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
        
    return valid_times

def cross_validation_performance_versust(data, reservoir, sync_length, train_length, pred_length, \
                                 pred_gap_length = 40, seed = 5, errormax = 3.2, train_method = 'RLS', progress = False):
    # Evaluates the valid time of prediction over a random set of training time series and test time series
    data_length = data.shape[0]
    train_lengths = np.arange(pred_gap_length, data_length - sync_length - pred_length, pred_gap_length)
    num_tests = train_lengths.size
    num_gaps = num_tests
    valid_times = np.zeros(num_tests)
    # For each starting point
    if progress: 
        with tqdm(total = num_gaps) as pbar:
            for i in range(num_tests):
                # Get validation data
                validation_data = data[sync_length+train_lengths[i]:sync_length+train_lengths[i]+pred_length]
                if i == 0:
                    train_data = data[:sync_length+train_lengths[i]] # Get training data

                    # Train the reservoir
                    if train_method == 'RLS':
                        reservoir.trainWoutRLS(train_data, sync_length)
                    else:
                        reservoir.trainWout(train_data, sync_length)
                else:
                    train_data = data[sync_length + train_lengths[i-1]:sync_length + train_lengths[i]] # Get training data
                    # Train the reservoir
                    if train_method == 'RLS':
                        reservoir.updateWoutRLS(train_data)
                    else:
                        reservoir.updateWout(train_data)
                # Evaluate the prediction
                valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
                pbar.update(1)
    else:
        for i in range(num_tests):
            # Get validation data
            validation_data = data[sync_length+train_lengths[i]:sync_length+train_lengths[i]+pred_length]
            if i == 0:
                train_data = data[:sync_length+train_lengths[i]] # Get training data
                
                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.trainWoutRLS(train_data, sync_length)
                else:
                    reservoir.trainWout(train_data, sync_length)
            else:
                train_data = data[sync_length + train_lengths[i-1]:sync_length + train_lengths[i]] # Get training data
                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.updateWoutRLS(train_data)
                else:
                    reservoir.updateWout(train_data)
            # Evaluate the prediction
            valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
        
    return valid_times

def cross_validation_performance_movingwindow(data, reservoir, sync_length, train_length, pred_length, \
                                 pred_gap_length = 40, seed = 5, errormax = 3.2, train_method = 'RLS', progress = False):
    # Evaluates the valid time of prediction over a random set of training time series and test time series
    data_length = data.shape[0]
    train_lengths = np.arange(pred_gap_length, data_length - sync_length - pred_length, pred_gap_length)
    num_tests = train_lengths.size
    num_gaps = num_tests
    valid_times = np.zeros(num_tests)
    # For each starting point
    if progress: 
        with tqdm(total = num_gaps) as pbar:
            for i in range(num_tests):
                # Get validation data
                validation_data = data[sync_length+train_lengths[i]:sync_length+train_lengths[i]+pred_length]
                if i == 0:
                    train_data = data[:sync_length+train_lengths[i]] # Get training data

                    # Train the reservoir
                    if train_method == 'RLS':
                        reservoir.trainWoutRLS(train_data, sync_length)
                    else:
                        reservoir.trainWout(train_data, sync_length)
                else:
                    train_data = data[train_lengths[i-1]:sync_length + train_lengths[i]] # Get training data
                    # Train the reservoir
                    if train_method == 'RLS':
                        reservoir.trainWoutRLS(train_data, sync_length)
                    else:
                        reservoir.trainWout(train_data, sync_length)
                # Evaluate the prediction
                valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
                pbar.update(1)
    else:
        for i in range(num_tests):
            # Get validation data
            validation_data = data[sync_length+train_lengths[i]:sync_length+train_lengths[i]+pred_length]
            if i == 0:
                train_data = data[:sync_length+train_lengths[i]] # Get training data

                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.trainWoutRLS(train_data, sync_length)
                else:
                    reservoir.trainWout(train_data, sync_length)
            else:
                train_data = data[train_lengths[i-1]:sync_length + train_lengths[i]] # Get training data
                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.trainWoutRLS(train_data, sync_length)
                else:
                    reservoir.trainWout(train_data, sync_length)
            # Evaluate the prediction
            valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax)
        
    return valid_times

def cross_validation_performance_separated(data, reservoir, num_tests, sync_length, train_length, pred_length, \
                                 errormax = 3.2, train_method = 'RLS', progress = False, plot = False, plot_idx = 0):
    # Evaluates the valid time of prediction over a random set of training time series and test time series
    data_length = data.shape[0]
    train_lengths = np.arange(0, data_length, train_length + sync_length + pred_length)
    if num_tests != train_lengths.size:
        raise ValueError
    num_gaps = num_tests
    valid_times = np.zeros(num_tests)
    # For each starting point
    if progress: 
        with tqdm(total = num_gaps) as pbar:
            for i in range(num_tests):
                # Get validation data
                validation_data = data[train_lengths[i] + sync_length + train_length:train_lengths[i] + pred_length + sync_length + train_length]
                train_data = data[train_lengths[i]:train_lengths[i] + train_length + sync_length] # Get training data
                
                # print((train_lengths[i], train_lengths[i] + train_length + sync_length))
                # Train the reservoir
                if train_method == 'RLS':
                    reservoir.trainWoutRLS(train_data, sync_length)
                else:
                    reservoir.trainWout(train_data, sync_length)
                # print((train_lengths[i] + sync_length + train_length, train_lengths[i] + pred_length + sync_length + train_length))
                # Evaluate the prediction
                valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax, plot = (plot and i == plot_idx))
                pbar.update(1)
    else:
        for i in range(num_tests):
            # Get validation data
            validation_data = data[train_lengths[i] + sync_length + train_length:train_lengths[i] + pred_length + sync_length + train_length]
            train_data = data[train_lengths[i]:train_lengths[i] + train_length + sync_length] # Get training data
            # print(train_data.shape[0])
            # print(validation_data.shape[0])
            # print((train_lengths[i], train_lengths[i] + train_length + sync_length))
            # Train the reservoir
            if train_method == 'RLS':
                reservoir.trainWoutRLS(train_data, sync_length)
            else:
                reservoir.trainWout(train_data, sync_length)
            # print((train_lengths[i] + sync_length + train_length, train_lengths[i] + pred_length + sync_length + train_length))
            # Evaluate the prediction
            valid_times[i]  = reservoir.valid_time(validation_data, error_bound = errormax, plot = (plot and i == plot_idx))
        
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

@jit(nopython = True, fastmath = True)
def updateTarget(data_trstates, aug_r, input, forget):
    data_trstates = forget*data_trstates + np.outer(input,aug_r)
    return data_trstates

@jit(nopython = True, fastmath = True)
def updateCovReg(states_trstates_inv, aug_r, inv_forget, regularization_mat, RLS_reg_type):
    states_trstates_inv = inv_forget*(states_trstates_inv - 1.0/\
        (1+inv_forget*np.dot(np.dot(aug_r,states_trstates_inv),aug_r))*\
        np.outer(np.dot(states_trstates_inv,aug_r),np.dot(aug_r,states_trstates_inv))\
        *inv_forget)
    if RLS_reg_type == 'LM':
        states_trstates_inv = states_trstates_inv - \
            regularization_mat * (states_trstates_inv @ states_trstates_inv)
    return states_trstates_inv
                                         
def updateTarget_tweighted(data_trstates, aug_r, input, forget, t_update_mat_T):
    data_trstates = forget * data_trstates @ t_update_mat_T + np.outer(input,aug_r)
    return data_trstates

def updateCovReg_tweighted(states_trstates_inv, aug_r, inv_forget, t_update_mat_inv, regularization_mat, RLS_reg_type):
    updated_states_trstates_inv = t_update_mat_inv.T @ states_trstates_inv @ t_update_mat_inv
    states_trstates_inv = updateCovReg(updated_states_trstates_inv, aug_r, inv_forget, regularization_mat, RLS_reg_type)

    return states_trstates_inv

def updateRes(r, leakage, A, Win, input):
    newr = leakage*r + (1-leakage)*\
            np.tanh(A.dot(r) + np.dot(Win,input))
    return newr


@jit(nopython = True, fastmath = True)
def computeWout(states, truth, forget, regularization):
    n = states.shape[0]
    d = states.shape[1]
    s_mat = np.power(forget,np.arange(n)[::-1])
    data_trstates = (truth.T * s_mat) @ states
    states_trstates = (states.T * s_mat) @ states
    WoutT = solve(states_trstates.T + regularization * np.identity(d), data_trstates.T)
    return WoutT.T, states, truth

@jit(nopython = True, fastmath = True)
def computeWoutTweighted(states, truth, forget, regularization,t_regularization, step):
    n,d = states.shape
    s_mat = np.power(forget,np.arange(n)[::-1])
    times = np.arange((1-n)*step,step,step)
    if np.abs(times[-1] - step)<1e-6:
        times = times[:times.size-1]
    # t_scale_mat = sparse.diags(times)
    reg_mat = np.append(regularization*np.ones(d), t_regularization*np.ones(d))
    aug_states = np.append(states, times.reshape(-1,1) *  states, axis = 1)
    data_trstates = (truth.T * s_mat) @ aug_states
    states_trstates = (aug_states.T * s_mat) @ aug_states + np.diag(reg_mat)
    WoutT = solve(states_trstates.T, data_trstates.T)
    return WoutT.T, states, truth