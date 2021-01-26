import numpy as np

def updateCovReg(states_trstates_inv, aug_r, inv_forget, regularization_mat, RLS_reg_type):
    states_trstates_inv = inv_forget*(states_trstates_inv - 1.0/\
        (1+inv_forget*np.dot(np.dot(aug_r,states_trstates_inv),aug_r))*\
        np.outer(np.dot(states_trstates_inv,aug_r),np.dot(aug_r,states_trstates_inv))\
        *inv_forget)
    if RLS_reg_type == 'LM':
        states_trstates_inv = states_trstates_inv - \
            regularization_mat * (states_trstates_inv @ states_trstates_inv)
    return states_trstates_inv