import os
import sys
import time
import numpy as np

import pytest

from context import linear_hf
from linear_hf.causality import causal_matrix
from linear_hf import neuralnet
# from linear_hf.causality import estimate_causality

@pytest.fixture
def make_nn_data():
    np.random.rand(1)

    # Make data with two markets: one growing exponentially
    # and one shrinking exponentially.
    n_time = 30
    n_markets = 2
    n_batch = 1

    data = np.random.rand(n_time, 2) * 1e-3 + 1

    data_all = np.array([np.hstack([data] * 4)])
    assert data_all.shape == (n_batch, n_time, n_markets * 4)
    return data_all

@pytest.fixture
def make_causal_data():
    fs = [lambda x: np.sin(x*10), 
          lambda x: np.cos(10*x),
          lambda x: -np.sin(10*x),
          lambda x: -np.cos(10*x)]
    xs = np.random.rand(1000, 4)
    ys = np.zeros((1000, 4))
    for y_id in range(4):
        ys[:, y_id] = fs[y_id](xs[:, y_id]) + np.random.rand(1000)

    return np.hstack([xs, ys])

def test_causal_matrix(make_causal_data):
    cm = causal_matrix(make_causal_data, method='nearest', n_neighbors=30, ind_method='hsic', thr=1e-2)
    should_be_causal = np.array([cm[0, 4], cm[1, 5], cm[2, 6], cm[3, 7],
                                 cm[0,0], cm[5,5], cm[3,3]])
    assert (should_be_causal < 1e-2).sum() == 0, 'Some of the causal relationships were not detected.'
    shouldnt_be_causal = np.array([cm[0,1], cm[1, 2], cm[2, 3], cm[3, 4],
                                  cm[4, 5], cm[5, 6], cm[6, 7], cm[0, 5],
                                  cm[1, 7], cm[2, 5], cm[5, 1], cm[6, 2]])
    assert (shouldnt_be_causal > 1e-2).sum() == 0, 'Some of the causal relationships were not detected.'

def test_gradient_decreases_loss_100steps(make_nn_data):
    np.random.rand(1)

    # Prepare the data.
    data_all = make_nn_data
    n_sharpe = 4
    batch_in = data_all[:, :-1, :]
    batch_out = data_all[:, -n_sharpe:, :]

    n_batch, n_time, n_ftrs = batch_in.shape
    horizon = n_time - n_sharpe + 1
    n_marketst4 = batch_out.shape[-1]
    n_markets = n_marketst4 / 4
    
    # Create a fake causality matrix that says, "the first market
    # causes the second market, at 0-timestep delay".
    cm = np.zeros((n_markets, n_markets))
    cm[0, 1] = 2.
    cm = np.tile(cm, [4, 1])

    # Compile a neural net that uses the causality matrix.
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, 
                          n_sharpe, lbd=1000., causality_matrix=cm)
    # Train the neural net for 10 steps and compute the loss.
    nn_l1_before = nn.l1_penalty_np()

    l1s = []
    for step_id in range(1000):
        nn.train_step(lr=1e-7, batch_in=batch_in, batch_out=batch_out)
        l1s.append(nn.l1_penalty_np())
    nn_l1_after = nn.l1_penalty_np()

    # Check that the causally-relevant weight is largest.
    W = nn.get_weights()
    W = W.reshape((horizon, n_ftrs, n_markets))
    w_csl = np.abs(W[:, 0, 1]).mean()
    w_noncsl = np.abs(W[:, 1, 0]).mean()
    assert nn_l1_before > nn_l1_after
    assert w_csl > w_noncsl
