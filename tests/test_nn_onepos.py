""" Test the one-position neural network cost,
which applies the same position vector (learned at Iteration 1)
to all the timesteps afterwards. """
import os
import sys
import pytest

import numpy as np
import tensorflow as tf

from context import linear_hf
from linear_hf.costs import compute_numpy_onepos_sharpe

@pytest.fixture
def make_data():
    n_batch = 17
    n_markets = 11
    n_time = 33
    n_sharpe = 7
    batch_in = np.random.rand(n_batch, n_time, n_markets * 4) + 5
    batch_out = batch_in + .01
    # batch_out = np.random.rand(n_batch, n_time, n_markets * 4)
    return batch_in, batch_out

def test_only_first_position_influences_sharpe(make_data):
    batch_in, batch_out = make_data
    n_batch, n_time, n_ftrs = batch_in.shape
    n_sharpe = n_time - 5
    n_markets = n_ftrs / 4
    horizon = n_time - n_sharpe + 1
    positions1 = np.random.rand(n_batch, n_sharpe, n_markets) - .5
    positions1 /= np.abs(positions1.sum(axis=2, keepdims=True))
    positions2 = np.random.rand(n_batch, n_sharpe, n_markets) - .5
    positions2 /= np.abs(positions2.sum(axis=2, keepdims=True))
    positions3 = np.array(positions2) - .5
    positions3 /= np.abs(positions3.sum(axis=2, keepdims=True))
    positions3[:, :1, :] = positions1[:, :1, :]

    sharpe1 = compute_numpy_onepos_sharpe(
        positions1, batch_out[:, -n_sharpe:])
    sharpe2 = compute_numpy_onepos_sharpe(
        positions2, batch_out[:, -n_sharpe:])
    sharpe3 = compute_numpy_onepos_sharpe(
        positions3, batch_out[:, -n_sharpe:])

    import pdb; pdb.set_trace()
    assert sharpe1 != sharpe2
    assert sharpe2 != sharpe3
    assert sharpe1 == sharpe3
