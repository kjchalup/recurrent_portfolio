import itertools

import numpy as np
from matplotlib import pyplot as plt

from . import iscause_anm

def causal_matrix(all_data, thr=1e-2):
    """ Compute the matrix of causal coefficients of the data.

    Args:
      all_data (n_samples, n_data): Data matrix.
      thr (scalar): the threshold at which to decide wheter a p-value passes the test.

    Returns:
      causal_coeffs (n_data, n_data): Matrix of such that causal_coeffs[i, j]
        is the p-value that i causes j if ij_pval > thr and ji_pval < thr. Otherwise, 0.
        In addition, the diagonal entries are all 1 (each variable "causes" itself).
    """
    n_data = all_data.shape[1]
    causal_coeffs = np.zeros((n_data, n_data))
    for x_id, y_id in itertools.product(range(n_data), repeat=2):
        if x_id == y_id:
            causal_coeffs[x_id, y_id] = 1
        elif x_id < y_id:
            xy_pval, yx_pval = iscause_anm(all_data[:, x_id:x_id+1], 
                                           all_data[:, y_id:y_id+1])
            if xy_pval > thr and yx_pval < thr:
                causal_coeffs[x_id, y_id] = xy_pval
            elif yx_pval > thr and xy_pval < thr:
                causal_coeffs[y_id, x_id] = yx_pval
    return causal_coeffs
