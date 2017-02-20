from scipy.spatial.distance import pdist, squareform
import numpy as np
import random
import copy

def distcorr(Xval, Yval, pval=True, nruns=100, max_data=500, **kwargs):
    """ Compute the distance correlation function, returning the p-value.
    Based on Satra/distcorr.py (gist aa3d19a12b74e9ab7941)
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    (0.76267624241686671, 0.268)
    """
    n_samples = Xval.size
    ids_to_use = np.random.permutation(n_samples)[:max_data]
    X = np.atleast_1d(Xval[ids_to_use])
    Y = np.atleast_1d(Yval[ids_to_use])
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    if pval:
        greater = 0
        for i in range(nruns):
            Y_r = copy.copy(Yval)
            random.shuffle(Y_r)
            if distcorr(Xval, Y_r, pval=False)[0] > dcor:
                greater += 1
        return (dcor, greater/float(n))
    else:
        return (dcor, dcor)
