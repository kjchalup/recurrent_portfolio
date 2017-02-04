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
import neuralnet

@pytest.fixture
def make_data():
    n_batch = 17
    n_markets = 11
    n_time = 33
    n_sharpe = 7
    batch_in = np.ones((n_batch, n_time, n_markets * 4), 
                       dtype=np.float32)
    return batch_in, n_sharpe

# def test_nn_all_inputs_ones(make_data):
#     batch_in, n_sharpe = make_data
#     n_batch, n_time, n_ftrs = batch_in.shape
#     n_markets = n_ftrs / 4
#     horizon = n_time - n_sharpe + 1
#     W_init = np.ones((n_ftrs * horizon, n_markets), dtype=np.float32)
#     nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, W_init)
#     assert_array_almost_equal(nn.predict(batch_in[:, -horizon:]), 
#                               np.ones((n_batch, n_markets)) / 
#                               float(n_markets))

# def test_nn_only_one_nonzero_data(make_data):
#     batch_in, n_sharpe = make_data
#     n_batch, n_time, n_ftrs = batch_in.shape
#     n_markets = n_ftrs / 4
#     batch_in[:, :, 1:] = 0
#     horizon = n_time - n_sharpe + 1
#     n_batch, n_time, n_markets = batch_in.shape
#     W_init = np.ones((n_ftrs * horizon, n_markets), dtype=np.float32)
#     nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, W_init)
#     assert_array_almost_equal(nn.predict(batch_in[:, -horizon:]), 
#                               np.ones((n_batch, n_markets)) / 
#                               float(n_markets))

# def test_nn_all_inputs_minus_ones(make_data):
#     batch_in, n_sharpe = make_data
#     n_batch, n_time, n_ftrs = batch_in.shape
#     horizon = n_time - n_sharpe + 1
#     n_markets = n_ftrs / 4
#     W_init = -np.ones((n_ftrs * horizon, n_markets), dtype=np.float32)
#     nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, W_init)
#     assert_array_almost_equal(nn.predict(batch_in[:, -horizon:]), 
#                               -np.ones((n_batch, n_markets)) / 
#                               float(n_markets))

# def test_nn_batch_order(make_data):
#     batch_in, n_sharpe = make_data
#     batch_in[0] = -batch_in[0]
#     n_batch, n_time, n_ftrs = batch_in.shape
#     horizon = n_time - n_sharpe + 1
#     n_markets = n_ftrs / 4
#     W_init = np.ones((n_ftrs * horizon, n_markets), dtype=np.float32)
#     nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, W_init)
#     out_first = nn.predict(batch_in[:, -horizon:])[0]
#     out_last = nn.predict(batch_in[:, -horizon:])[-1]
#     assert_array_almost_equal(
#         out_first, -np.ones(n_markets) / float(n_markets))
#     assert_array_almost_equal(
#         out_last, np.ones(n_markets) / float(n_markets))

def test_nn_positions(make_data):
    batch_in, n_sharpe = make_data
    batch_in[0] = -batch_in[0]
    n_batch, n_time, n_ftrs = batch_in.shape
    horizon = n_time - n_sharpe + 1
    n_markets = n_ftrs / 4
    W_init = np.ones((n_ftrs * horizon, n_markets), dtype=np.float32)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, W_init)
    pos = nn._positions_np(batch_in)
    assert 1==0
    
    
