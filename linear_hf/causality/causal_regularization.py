import itertools
from joblib import Parallel, delayed

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import iscause_anm

def causal_matrix(all_data, thr=0, verbose=False, method='nearest', **kwargs):
    """ Compute the matrix of causal coefficients of the data. Each 
    coefficient is the p-value of the casual direction.

    Args:
      all_data (n_samples, n_data): Data matrix.
      thr (scalar): The threshold at which to decide wheter a p-value passes the test.
      verbose: If True, display messages and save plots to a file.
      method: Regression method to use. Can be 'poly' or 'nearest'.
      kwargs: Arguments for the regression method.

    Returns:
      causal_coeffs (n_data, n_data): Matrix of such that causal_coeffs[i, j]
        is the p-value that i causes j if ij_pval > thr and ji_pval <= thr. Otherwise, 0.
        In addition, the diagonal entries are all 1 (each variable "causes" itself).
    """
    n_data = all_data.shape[1]
    causal_coeffs = np.zeros((n_data, n_data))
    iter_id = 0
    if verbose:
        pdf = PdfPages('saved_data/causal_matrix.pdf')
    for x_id, y_id in itertools.product(range(n_data), repeat=2):
        if x_id == y_id:
            causal_coeffs[x_id, y_id] = 1
        elif x_id < y_id:
            fig = plt.figure(figsize=(10,5)) if verbose else None
            xy_pval, yx_pval = iscause_anm(
                all_data[:, x_id:x_id+1], all_data[:, y_id:y_id+1], 
                fig=fig, method=method, pval=True, **kwargs)
            if xy_pval > thr and yx_pval <= thr:
                causal_coeffs[x_id, y_id] = xy_pval
            elif yx_pval > thr and xy_pval <= thr:
                causal_coeffs[y_id, x_id] = yx_pval
            print(('Computed causality {}/{} [xid={}, yid={}]. '
                   'xy_pval = {:.4g}, yx_pval = {:.4g}').format(
                iter_id, n_data**2, x_id, y_id, xy_pval, yx_pval))
            if verbose:
                pdf.savefig(fig)
                plt.close()

        iter_id += 1
    if verbose:
        pdf.close()
    return causal_coeffs

def causal_matrix_ratios(all_data, thr=0, verbose=False, method='nearest', **kwargs):
    """ Compute the matrix of causal coefficients of the data, where each
    coefficient is the ratio of the distance-correlation of the two
    causal directions.

    Args:
      all_data (n_samples, n_data): Data matrix.
      thr (scalar): The threshold at which to decide wheter a p-value passes the test.
      verbose: If True, display messages and save plots to a file.
      method: Regression method to use. Can be 'poly' or 'nearest'.
      kwargs: Arguments for the regression method.

    Returns:
      causal_coeffs (n_data, n_data): Matrix of such that causal_coeffs[i, j]
        is the p-value that i causes j if ij_pval > thr and ji_pval <= thr. Otherwise, 0.
        In addition, the diagonal entries are all 1 (each variable "causes" itself).
    """
    n_data = all_data.shape[1]
    causal_coeffs = np.zeros((n_data, n_data))
    iter_id = 0
    if verbose:
        pdf = PdfPages('saved_data/causal_matrix.pdf')
    for x_id, y_id in itertools.product(range(n_data), repeat=2):
        if x_id == y_id:
            causal_coeffs[x_id, y_id] = 1.
        elif x_id < y_id:
            fig = plt.figure(figsize=(10,5)) if verbose else None
            xy_pval, yx_pval = iscause_anm(
                all_data[:, x_id:x_id+1], all_data[:, y_id:y_id+1],
                fig=fig, method=method, pval=False, **kwargs)
            if np.isnan(xy_pvals) or np.isnan(xy_pvals):
                xy_pval = 1.
                yx_pval = 1.
            causal_coeffs[x_id, y_id] = xy_pval / yx_pval
            causal_coeffs[y_id, x_id] = yx_pval / xy_pval
            print(('Computed causality {}/{} [xid={}, yid={}]. '
                   'xy_pval = {:.4g}, yx_pval = {:.4g}').format(
                iter_id, n_data**2, x_id, y_id, xy_pval, yx_pval))
            if verbose:
                pdf.savefig(fig)
                plt.close()

        iter_id += 1
    if verbose:
        pdf.close()
    return causal_coeffs
