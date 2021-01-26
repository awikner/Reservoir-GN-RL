import numpy as np
cimport numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from numpy.linalg import solve

DTYPE=np.double
ctypedef np.double_t DTYPE_t

def updateRes(r, leakage, A, Win, input):
    newr = leakage*r + (1-leakage)*\
            np.tanh(A.dot(r) + np.dot(Win,input))
    return newr

def updateCovReg(np.ndarray[DTYPE_t, ndim = 2] states_trstates_inv, np.ndarray[DTYPE_t, ndim = 1] aug_r, double inv_forget, \
                 np.ndarray[DTYPE_t, ndim = 2] regularization_mat, str RLS_reg_type):
    states_trstates_inv = inv_forget*(states_trstates_inv - 1.0/\
        (1+inv_forget*np.dot(np.dot(aug_r,states_trstates_inv),aug_r))*\
        np.outer(np.dot(states_trstates_inv,aug_r),np.dot(aug_r,states_trstates_inv))\
        *inv_forget)
    if RLS_reg_type == 'LM':
        states_trstates_inv = states_trstates_inv - \
            regularization_mat * (states_trstates_inv @ states_trstates_inv)
    return states_trstates_inv

def updateCovReg_tweighted(states_trstates_inv, aug_r, inv_forget, t_update_mat_inv, regularization_mat, RLS_reg_type):
    updated_states_trstates_inv = t_update_mat_inv.T @ states_trstates_inv @ t_update_mat_inv
    states_trstates_inv = updateCovReg(updated_states_trstates_inv, aug_r, inv_forget, regularization_mat, RLS_reg_type)
    return states_trstates_inv

def updateTarget(np.ndarray[DTYPE_t, ndim = 2] data_trstates, np.ndarray[DTYPE_t, ndim = 1] aug_r,\
                 np.ndarray[DTYPE_t, ndim = 1] input, double forget):
    data_trstates = forget*data_trstates + np.outer(input,aug_r)
    return data_trstates

def updateTarget_tweighted(data_trstates, aug_r, input, forget, t_update_mat_T):
    data_trstates = forget * data_trstates @ t_update_mat_T + np.outer(input,aug_r)
    return data_trstates