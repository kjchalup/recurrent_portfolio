import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

from kerpy.GaussianKernel import GaussianKernel
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject

def compute_independence(x, y):
    """ Use the Hilbert-Schmidt Independence Criterion to decide whether
    samples in x and y come from statistically independent distributions.

    Args:
      x (n_samples, n_dim): Samples.
      y (n_samples, n_dim): Samples.

    Returns:
      pval (scalar): a p-value that should be high if the distributions are independent.
    """
    n_samples = x.shape[0]
    kernel_x = GaussianKernel() 
    kernel_y = GaussianKernel()
    test_object = HSICSpectralTestObject(n_samples, kernelX=kernel_x, 
                                 kernelY=kernel_y, 
                                 kernelX_use_median=True,
                                 kernelY_use_median=True,
                                 rff=True, num_rfx=20, num_rfy=20,
                                 num_nullsims=1000)
    return test_object.compute_pvalue(x, y)

def iscause_anm(x, y, n_neighbors=30):
    """ Decide whether x causes y using the additive noise models idea.

    Args:
      x (n_samples, n_ftrs_x): Input data.
      y (n_samples, n_ftrs_y): Output data.

    Returns:
      xy_pval: A p-value that should be large if x causes y.
      yx_pval: A p-value that should be large if y causes x.
    """

    clf = KNeighborsRegressor(n_neighbors=n_neighbors)
    clf.fit(x, y.flatten())
    y_pred = clf.predict(x).reshape(-1, 1)
    noise_xy = y - y_pred
    xy_pval = compute_independence(x, noise_xy)

    clf.fit(y, x.flatten())
    x_pred = clf.predict(y).reshape(-1, 1)
    noise_yx = x - x_pred
    yx_pval = compute_independence(y, noise_yx)
    return xy_pval, yx_pval
