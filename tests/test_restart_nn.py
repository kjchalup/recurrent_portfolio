import pytest
import numpy as np
from context import linear_hf
from linear_hf import neuralnet
from linear_hf import chunknet
from linear_hf.run_backtest import restart_nn_till_good
from linear_hf.run_backtest import loss_calc

def test_nnlinear_restart():
    n_time = 100
    n_markets = 1

    data = np.random.rand(n_time, 1) + 10
    data2 = np.random.rand(n_time, 1) + 1

    data_out = np.hstack([data, data2] * 4)
    data_in = np.hstack([data, data2])

    n_ftrs = 2
    n_time = 30
    n_sharpe = 5
    n_markets = 2
    neural = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': neural,
                'n_time': n_time,
                'n_sharpe': n_sharpe,
                'horizon': n_time - n_sharpe + 1,
                'lr': 1e-3,
                'batch_size': 5}
    settings['val_period'] = 5

    pos_b4 = settings['nn'].predict(data_in[:settings['horizon']])


    settings, all_val, market_val, all_batch, market_batch, best_sharpe = restart_nn_till_good(
        settings=settings,
        num_times=100,
        all_data=data_in,
        market_data=data_out,
        debug=True)
    calc_loss = loss_calc(settings, all_batch=all_val, market_batch=market_val)
    pos_after = settings['nn'].predict(data_in[:settings['horizon']])

    assert calc_loss == best_sharpe
    assert not (pos_b4 == pos_after).any(), "NN weights are still the same after restart!"

    settings['val_period'] = 0
    settings, all_val, market_val, all_batch, market_batch, best_sharpe = restart_nn_till_good(
        settings=settings,
        num_times=5,
        all_data=data_in,
        market_data=data_out,
        debug=True)
    calc_loss = loss_calc(settings, all_batch=all_batch, market_batch=market_batch)
    assert calc_loss == best_sharpe
    assert not (pos_b4 == pos_after).any(), "NN weights are still the same after restart!"
def test_chunknet_restart():
    n_time = 100
    n_markets = 1

    data = np.random.rand(n_time, 1) + 10
    data2 = np.random.rand(n_time, 1) + 1

    data_out = np.hstack([data, data2] * 4)
    data_in = np.hstack([data, data2])

    n_ftrs = 2
    n_time = 30
    n_sharpe = 5
    n_markets = 2
    neural = chunknet.ChunkLinear(n_ftrs, n_markets, n_time, n_sharpe,
                                  n_chunks=2)
    settings = {'nn': neural,
                'n_time': n_time,
                'n_sharpe': n_sharpe,
                'horizon': n_time - n_sharpe + 1,
                'lr': 1e-3,
                'batch_size': 5}
    settings['val_period'] = 5

    pos_b4 = settings['nn'].predict(data_in[:settings['horizon']])


    settings, all_val, market_val, all_batch, market_batch, best_sharpe = restart_nn_till_good(
        settings=settings,
        num_times=100,
        all_data=data_in,
        market_data=data_out,
        debug=True)
    calc_loss = loss_calc(settings, all_batch=all_val, market_batch=market_val)
    pos_after = settings['nn'].predict(data_in[:settings['horizon']])

    assert calc_loss == best_sharpe
    assert not (pos_b4 == pos_after).any(), "NN weights are still the same after restart!"

    settings['val_period'] = 0
    settings, all_val, market_val, all_batch, market_batch, best_sharpe = restart_nn_till_good(
        settings=settings,
        num_times=5,
        all_data=data_in,
        market_data=data_out,
        debug=True)
    calc_loss = loss_calc(settings, all_batch=all_batch, market_batch=market_batch)
    assert calc_loss == best_sharpe
    assert not (pos_b4 == pos_after).any(), "NN weights are still the same after restart!"
#
