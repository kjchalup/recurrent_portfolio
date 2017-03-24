import pytest
import numpy as np
from numpy.testing import assert_array_equal
from context import rnn_portfolio
from rnn_portfolio import rnn
from rnn_portfolio import NP_DTYPE
from rnn_portfolio.costs import compute_np_sharpe
from rnn_portfolio import training
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def make_data():
    n_batch = 17
    n_markets = 11
    n_time = 33
    n_sharpe = 7
    batch_in = np.ones((n_batch, n_time, n_markets * 4),
                       dtype=NP_DTYPE)
    return batch_in, n_sharpe


def test_prediction_onestep():
    """ Test if the rnn correctly saves states predicting one-day-at-a-time. """
    np.random.rand(1)
    n_time = 50
    n_markets = 3

    # Make up some fake data.
    data1 = np.random.rand(n_time * 10, 1) + 20
    data2 = np.random.rand(n_time * 10, 1) + 5
    data3 = np.cumprod(np.ones((n_time * 10, 1)) * 1.01, axis=0)
    data_all = np.hstack([data1, data2, data3] * 4)
    data_in = StandardScaler().fit_transform(np.hstack([data1, data2, data3]))
    
    n_sharpe = 40
    n_ftrs = data_in.shape[1]
    net = rnn.RNN(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': net,
                'nn_type': 'rnn',
                'num_epochs': 2,
                'n_time': n_time,
                'n_sharpe': n_sharpe,
                'horizon':  n_time - n_sharpe + 1,
                'val_period': 0,
                'lr': 1e-2,
                'lr_mult_base': 1.,
                'batch_size': 5,
                'iter': 0,
                'lbd': 0,
                'realized_sharpe': [],
                'saved_val_sharpe': [],
                'retrain_interval': 1,
                'allow_shorting': True}
    
    settings = training.train(settings, all_data=data_in, market_data=data_all)
    nn_pos1 = settings['nn']._positions_np(batch_in=data_in[None, :n_time])
    nn_pos2 = settings['nn']._positions_np(batch_in=data_in[None, :n_time])
    assert_array_equal(nn_pos1, nn_pos2, 'Predictions on same data differ.')
    nn_onestep = np.array([settings['nn'].predict(
        data_in=data_in[start_id:start_id+1]) for start_id in range(n_time)])
    assert_array_equal(nn_pos1[0], nn_onestep, 'Onestep predictions differ.')


# def test_training_longing():
#     """ Test if the rnn longs an exponentially-increasing stock. """
#     np.random.rand(1)
#     n_time = 50
#     n_markets = 3
#     n_sharpe = 10

#     # Make up some fake data.
#     data1 = np.random.rand(n_time, 1) + 20
#     data2 = np.random.rand(n_time, 1) + 5
#     data3 = np.cumprod(np.ones((n_time, 1)) * 1.01, axis=0)
#     data_all = np.hstack([data1, data2, data3] * 4)
#     data_in = StandardScaler().fit_transform(np.hstack([data1, data2, data3]))
    

#     n_ftrs = data_in.shape[1]
#     n_time -= 20
#     net = rnn.RNN(n_ftrs, n_markets, n_time, n_sharpe)
#     settings = {'nn': net,
#                 'nn_type': 'rnn',
#                 'num_epochs': 50,
#                 'n_time': n_time,
#                 'n_sharpe': n_sharpe,
#                 'horizon':  n_time - n_sharpe + 1,
#                 'val_period': 0,
#                 'lr': 1e-2,
#                 'lr_mult_base': 1.,
#                 'batch_size': 5,
#                 'iter': 0,
#                 'lbd': 0,
#                 'realized_sharpe': [],
#                 'saved_val_sharpe': [],
#                 'retrain_interval': 1,
#                 'allow_shorting': True}
    
#     settings = training.train(settings, all_data=data_in, market_data=data_all)
#     nn_pos = settings['nn']._positions_np(batch_in=data_in[None, :n_time, :])
#     sharpe = compute_np_sharpe(positions=nn_pos, prices=data_all[None, :n_time, :])
#     net.sess.close()
#     assert sharpe > 20


# def test_training_shorting():
#     """ Test if the net learns to short an exponentially-decreasing stock. """
#     np.random.rand(1)
#     n_time = 50
#     n_markets = 3
#     n_sharpe = 10

#     # Make up some fake data.
#     data1 = np.random.rand(n_time, 1) + 20
#     data2 = np.random.rand(n_time, 1) + 5
#     data3 = np.cumprod(np.ones((n_time, 1)) * .99, axis=0)
#     data_all = np.hstack([data1, data2, data3] * 4)
#     data_in = StandardScaler().fit_transform(np.hstack([data1, data2, data3]))

#     n_ftrs = data_in.shape[1]
#     n_time = data_all.shape[0] - 20
#     net = rnn.RNN(n_ftrs, n_markets, n_time, n_sharpe)
#     settings = {'nn': net,
#                 'nn_type': 'rnn',
#                 'num_epochs': 50,
#                 'n_time': n_time,
#                 'n_sharpe': n_sharpe,
#                 'horizon': n_time - n_sharpe + 1,
#                 'val_period': 0,
#                 'lr': 1e-2,
#                 'lr_mult_base': 0.1,
#                 'batch_size': 5,
#                 'iter': 0,
#                 'lbd': 0,
#                 'realized_sharpe': [],
#                 'saved_val_sharpe': [],
#                 'retrain_interval': 1,
#                 'allow_shorting': True}

#     settings = training.train(settings, all_data=data_in, market_data=data_all)
#     nn_pos = settings['nn']._positions_np(batch_in=data_in[None, :n_time, :])
#     sharpe = compute_np_sharpe(positions=nn_pos, prices=data_all[None, :n_time, :])
#     net.sess.close()
#     assert sharpe > 20


