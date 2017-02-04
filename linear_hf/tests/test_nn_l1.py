import pytest
import os
import sys
import inspect 

#Include scripts from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal
import neuralnet

def test_nn_l1_positive_weights():
    n_markets = 10
    n_time = 7
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1

    W_init = np.ones((n_markets * 4 * horizon, n_markets), 
                     dtype=np.float32)
    nn = neuralnet.Linear(n_markets * 4, n_markets, 
                          n_time, n_sharpe, W_init)
    assert nn.l1_penalty_np(lbd=1.) == np.abs(W_init).sum()

def test_nn_l1_negative_weights():
    n_markets = 10
    n_time = 7
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1

    W_init = -np.ones((n_markets * 4 * horizon, n_markets), 
                      dtype=np.float32)
    nn = neuralnet.Linear(n_markets * 4, n_markets, 
                          n_time, n_sharpe, W_init)
    assert nn.l1_penalty_np(lbd=1.) == np.abs(W_init).sum()

def test_nn_l1_mixed_weights():
    n_markets = 10
    n_time = 7
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1

    W_init = np.ones((n_markets * 4 * horizon, n_markets), 
                     dtype=np.float32)
    W_init[np.random.rand(*W_init.shape) > .5] *= -1
    nn = neuralnet.Linear(n_markets * 4, n_markets, 
                          n_time, n_sharpe, W_init)
    assert nn.l1_penalty_np(lbd=1.) == np.abs(W_init).sum()

def test_nn_l1_fractional_weights():
    n_markets = 10
    n_time = 7
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1

    W_init = np.ones((n_markets * 4 *  horizon, n_markets), 
                     dtype=np.float32) 
    W_init *= 1e-10
    nn = neuralnet.Linear(n_markets * 4, n_markets, 
                          n_time, n_sharpe, W_init)
    assert_almost_equal(nn.l1_penalty_np(lbd=1.), np.abs(W_init).sum())
