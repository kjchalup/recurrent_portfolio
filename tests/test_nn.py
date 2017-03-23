import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from context import linear_hf
from linear_hf import neuralnet
from linear_hf import NP_DTYPE
from linear_hf.costs import compute_np_sharpe
from linear_hf import training

@pytest.fixture
def make_data():
    n_batch = 17
    n_markets = 11
    n_time = 33
    n_sharpe = 7
    batch_in = np.ones((n_batch, n_time, n_markets * 4),
                       dtype=NP_DTYPE)
    return batch_in, n_sharpe


def test_nn_all_inputs_ones(make_data):
    batch_in, n_sharpe = make_data
    _, n_time, n_ftrs = batch_in.shape
    n_markets = n_ftrs / 4
    horizon = n_time - n_sharpe + 1
    w_init = np.ones((n_ftrs * horizon, n_markets), dtype=NP_DTYPE)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, w_init)
    assert_array_almost_equal(nn.predict(batch_in[0, -horizon:]),
                              np.ones(n_markets) / float(n_markets))


def test_nn_only_one_nonzero_data(make_data):
    batch_in, n_sharpe = make_data
    _, n_time, n_ftrs = batch_in.shape
    n_markets = n_ftrs / 4
    batch_in[:, :, 1:] = 0
    horizon = n_time - n_sharpe + 1
    n_batch, n_time, n_markets = batch_in.shape
    w_init = np.ones((n_ftrs * horizon, n_markets), dtype=NP_DTYPE)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, w_init)
    assert_array_almost_equal(nn.predict(batch_in[0, -horizon:]),
                              np.ones(n_markets) / float(n_markets))


def test_nn_all_inputs_minus_ones(make_data):
    batch_in, n_sharpe = make_data
    _, n_time, n_ftrs = batch_in.shape
    horizon = n_time - n_sharpe + 1
    n_markets = n_ftrs / 4
    w_init = -np.ones((n_ftrs * horizon, n_markets), dtype=NP_DTYPE)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, w_init)
    assert_array_almost_equal(nn.predict(batch_in[0, -horizon:]),
                              -np.ones(n_markets) / float(n_markets))


def test_nn_batch_order(make_data):
    batch_in, n_sharpe = make_data
    batch_in[0] = -batch_in[0]
    _, n_time, n_ftrs = batch_in.shape
    horizon = n_time - n_sharpe + 1
    n_markets = n_ftrs / 4
    w_init = np.ones((n_ftrs * horizon, n_markets), dtype=NP_DTYPE)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, w_init)
    out_first = nn.predict(batch_in[0, -horizon:])
    out_last = nn.predict(batch_in[-1, -horizon:])
    assert_array_almost_equal(
        out_first, -np.ones(n_markets) / float(n_markets))
    assert_array_almost_equal(
        out_last, np.ones(n_markets) / float(n_markets))


def test_nn_positions(make_data):
    batch_in, n_sharpe = make_data
    batch_in[0] = -batch_in[0]
    n_batch, n_time, n_ftrs = batch_in.shape
    horizon = n_time - n_sharpe + 1
    n_markets = n_ftrs / 4
    w_init = np.ones((n_ftrs * horizon, n_markets), dtype=NP_DTYPE)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, w_init)
    pos = nn._positions_np(batch_in)
    assert pos.shape == (n_batch, n_sharpe, n_markets)

def test_training_longing():
    """ Test if the net longs an exponentially-increasing stock. """
    np.random.rand(1)
    n_time = 50
    n_markets = 3
    n_sharpe = 10

    # Make up some fake data.
    data1 = np.random.rand(n_time, 1) + 20
    data2 = np.random.rand(n_time, 1) + 5
    data3 = np.cumprod(np.ones((n_time, 1)) * 1.01, axis=0)
    data_all = np.hstack([data1, data2, data3] * 4)
    data_in = np.hstack([data1, data2, data3])

    n_ftrs = data_in.shape[1]
    n_time = data_all.shape[0] - 20
    neural = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': neural,
                'num_epochs': 50,
                'n_time': n_time,
                'n_sharpe': n_sharpe,
                'horizon': n_time - n_sharpe + 1,
                'val_period': 0,
                'lr': 1e-3,
                'lr_mult_base': 0.1,
                'batch_size': 5,
                'iter': 0,
                'lbd': 0,
                'realized_sharpe': [],
                'saved_val_sharpe': [],
                'retrain_interval': 1,
                'nn_type': 'rnn',
                'allow_shorting': True}

    settings = training.train(settings, all_data=data_in, market_data=data_all)
    nn_pos = settings['nn']._positions_np(batch_in=data_in[None, :n_time, :])
    sharpe = compute_np_sharpe(positions=nn_pos, prices=data_all[None, :n_time, :])
    assert sharpe > 20


def test_training_shorting():
    """ Test if the net learns to short an exponentially-decreasing stock. """
    np.random.rand(1)
    n_time = 50
    n_markets = 3
    n_sharpe = 10

    # Make up some fake data.
    data1 = np.random.rand(n_time, 1) + 20
    data2 = np.random.rand(n_time, 1) + 5
    data3 = np.cumprod(np.ones((n_time, 1)) * .99, axis=0)
    data_all = np.hstack([data1, data2, data3] * 4)
    data_in = np.hstack([data1, data2, data3])

    n_ftrs = data_in.shape[1]
    n_time = data_all.shape[0] - 20
    neural = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': neural,
                'num_epochs': 50,
                'n_time': n_time,
                'n_sharpe': n_sharpe,
                'horizon': n_time - n_sharpe + 1,
                'val_period': 0,
                'lr': 1e-3,
                'lr_mult_base': 0.1,
                'batch_size': 5,
                'iter': 0,
                'lbd': 0,
                'realized_sharpe': [],
                'saved_val_sharpe': [],
                'nn_type': 'rnn',
                'retrain_interval': 1,
                'allow_shorting': True}

    settings = training.train(settings, all_data=data_in, market_data=data_all)
    nn_pos = settings['nn']._positions_np(batch_in=data_in[None, :n_time, :])
    sharpe = compute_np_sharpe(positions=nn_pos, prices=data_all[None, :n_time, :])
    assert sharpe > 20
