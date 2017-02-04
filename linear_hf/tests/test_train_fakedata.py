import pytest
import os
import sys
import inspect 
#Include scripts from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np

from preprocessing import load_stocks
import neuralnet
import costs


@pytest.fixture
def make_data():
    np.random.rand(1)

    # Make data with two markets: one growing exponentially
    # and one shrinking exponentially.
    n_time = 30
    n_markets = 2
    n_batch = 1

    data1 = np.ones((n_time, 1)) * 1.1
    data1 = np.cumprod(data1, axis=0)

    data2 = np.ones((n_time, 1)) * .9
    data2 = np.cumprod(data2, axis=0)

    data_all = np.array([np.hstack([data1, data2] * 4)])
    assert data_all.shape == (n_batch, n_time, n_markets * 4)
    return data_all

def test_gradient_decreases_loss_1step(make_data):
    np.random.rand(1)

    data_all = make_data
    n_sharpe = 4
    batch_in = data_all[:, :-1, :]
    batch_out = data_all[:, -n_sharpe:, :]

    n_batch, n_time, n_ftrs = batch_in.shape
    horizon = n_time - n_sharpe + 1
    n_marketst4 = batch_out.shape[-1]
    n_markets = n_marketst4 / 4
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=0)
    nn_loss_before = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn.train_step(lr=1e-7, batch_in=batch_in, batch_out=batch_out)
    nn_loss_after = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    assert nn_loss_before > nn_loss_after

# def test_gradient_decreases_loss_100steps(make_data):
#     np.random.rand(1)

#     data_all = make_data
#     n_sharpe = 4
#     batch_in = data_all[:, :-1, :]
#     batch_out = data_all[:, -n_sharpe:, :]

#     n_batch, n_time, n_ftrs = batch_in.shape
#     horizon = n_time - n_sharpe + 1
#     n_marketst4 = batch_out.shape[-1]
#     n_markets = n_marketst4 / 4
#     nn = neuralnet.Linear(n_ftrs, n_markets, n_time, 
#                           n_sharpe, lbd=0)
#     nn_loss_before = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
#     for step_id in range(10):
#         nn.train_step(lr=1e-5, batch_in=batch_in, batch_out=batch_out)
#     nn_loss_after = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
#     positions = nn._positions_np(batch_in)
#     import pdb; pdb.
#     assert positions[:, 0].mean() > positions[:, 1].mean()
