import numpy as np
from scipy.sparse import coo_matrix
def updateRes(r, leakage, A, Win, input):
    newr = leakage*r + (1-leakage)*\
            np.tanh(A.dot(r) + np.dot(Win,input))
    return newr