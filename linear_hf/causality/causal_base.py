import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt

from tools.distcorr import distcorr
from kerpy.GaussianKernel import GaussianKernel
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject

def compute_independence(x, y, ind_method='correlation', **kwargs):
    """ Use the Hilbert-Schmidt Independence Criterion to decide whether
    samples in x and y come from statistically independent distributions.

    Args:
      x (n_samples, n_dim): Samples.
      y (n_samples, n_dim): Samples.
      ind_method: Independence testing method to use. 'hsic' or 'correlation'.

    Returns:
      pval (scalar): a p-value that should be high if the distributions are independent.
    """
    n_samples = x.shape[0]
    if ind_method=='hsic':
        kernel_x = GaussianKernel() 
        kernel_y = GaussianKernel()
        test_object = HSICSpectralTestObject(n_samples, kernelX=kernel_x, 
                                     kernelY=kernel_y, 
                                     kernelX_use_median=True,
                                     kernelY_use_median=True,
                                     rff=True, num_rfx=20, num_rfy=20,
                                     num_nullsims=1000)
        pval = test_object.compute_pvalue(x, y)
    elif ind_method=='correlation':
        _, pval = distcorr(x.flatten(), y.flatten(), pval=True, **kwargs)
    return pval

def iscause_anm(x, y, fig=None, method='poly', **kwargs):
    """ Decide whether x causes y using the additive noise models idea.

    Args:
      x (n_samples, n_ftrs_x): Input data.
      y (n_samples, n_ftrs_y): Output data.
      fig: pyplot figure to plot to. Can be None (no plotting).
      method: Regression method to use. Can be 'poly', 'nearest'.
      kwargs: Arguments for the regression method.

    Returns:
      xy_pval: A p-value that should be large if x causes y.
      yx_pval: A p-value that should be large if y causes x.
    """
    
    if method=='nearest':
        clf = KNeighborsRegressor(kwargs['n_neighbors'])
        clf.fit(x, y.flatten())
        y_pred = clf.predict(x).reshape(-1, 1)
    elif method=='poly':
        z = np.polyfit(x.flatten(), y.flatten(), **kwargs)
        y_pred = np.poly1d(z)(x)
    noise_xy = y - y_pred
    xy_pval = compute_independence(x, noise_xy, **kwargs)

    if method=='nearest':
        clf.fit(y, x.flatten())
        x_pred = clf.predict(y).reshape(-1, 1)
    elif method=='poly':
        z = np.polyfit(y.flatten(), x.flatten(), **kwargs)
        x_pred = np.poly1d(z)(y)
    noise_yx = x - x_pred
    yx_pval = compute_independence(y, noise_yx, **kwargs)
    
    if fig is not None:
        ax=fig.add_subplot(221)
        ax.set_title('pval = {:.4g}'.format(xy_pval))
        ax.plot(x, y, 'k.', x, y_pred, 'r.')
        ax=fig.add_subplot(222)
        ax.plot(x, noise_xy, 'k.')
        ax=fig.add_subplot(223)
        ax.set_title('pval = {:.4g}'.format(yx_pval))
        ax.plot(y, x, 'k.', y, x_pred, 'r.')
        ax=fig.add_subplot(224)
        ax.plot(y, noise_yx, 'k.')


    return xy_pval, yx_pval
