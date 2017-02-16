import pytest
import quantiacsToolbox
import numpy as np

from context import linear_hf
from linear_hf.run_backtest import dont_trade_positions
from linear_hf.run_backtest import calc_batches
from linear_hf.run_backtest import lr_calc
from linear_hf.run_backtest import update_nn
from linear_hf.run_backtest import loss_calc

from linear_hf import  neuralnet
from linear_hf.costs import compute_numpy_sharpe
from linear_hf.run_backtest import training
def test_donttradepositions():
    positions=np.array([1, 1, 1])
    settings = {'markets': ['COL1','COL2','CASH']}
    positions = dont_trade_positions(positions, settings)
    assert positions[0] == 0
    assert positions[1] == 0
    assert positions[2] == 1

def test_calc_batches():
    settings = {'horizon': 51,
                'val_period': 50,
                'n_sharpe': 50,
                'batch_size': 50}
    n_timesteps = 1000
    
    batches_per_epoch = calc_batches(n_timesteps=n_timesteps, settings=settings)
    
    calc = (1000 - 2 * settings['n_sharpe'] - settings['horizon'] - settings['val_period'] + 1)
    calc = calc / float(settings['batch_size'])
    calc = int(calc)

    assert batches_per_epoch == calc
    assert batches_per_epoch == 16
    
    batches_per_epoch = calc_batches(n_timesteps=999, settings=settings)

    assert batches_per_epoch == calc-1

def test_lr_calc():
    settings = {'lr_mult_base': 0.1,
                'lr': 10,
                'num_epochs': 10}

    lr_new = lr_calc(settings, epoch_id=0)
    assert lr_new == settings['lr']

    lr_new = lr_calc(settings, epoch_id=10)
    ratio = lr_new / (settings['lr'] * settings['lr_mult_base'])
    assert ratio > 0.99 and ratio < 1.01
    
def test_loss_calc():
    np.random.rand(1)
    n_time = 20
    n_markets = 2
    n_batch = 1
    n_sharpe = 19

    data1 = np.random.rand(n_time, 1) + 20

    data2 = np.random.rand(n_time, 1) + 5

    data_all = np.array([np.hstack([data1, data2] *4)])
    assert data_all.shape == (n_batch, n_time, n_markets * 4)
    
    n_ftrs = data_all.shape[2]
    n_time = data_all.shape[1]
    n_sharpe = n_time - 1
    n_time = n_time - 1
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': nn}
    
    batch_in = data_all[:, :-1, :]
    batch_out = data_all[:, 1:, :]

    sharpe_nn = loss_calc(settings, all_batch=batch_in, market_batch=batch_out)
    nn_pos = nn._positions_np(batch_in=batch_in)
    sharpe_np = compute_numpy_sharpe(positions=nn_pos, prices=batch_out, slippage=0.05)
    ratio = sharpe_nn / sharpe_np
    assert ratio < 1.01 and ratio > 0.99

def test_update_nn():
    n_ftrs = 4
    n_markets = 2
    n_time = 10
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1
    
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': nn}
    best_sharpe = 10
    epoch_sharpe = 20
    settings, best_sharpe = update_nn(settings, best_sharpe, epoch_sharpe)

    assert best_sharpe == epoch_sharpe

    best_sharpe = 20
    epoch_sharpe = 10

    settings, best_sharpe = update_nn(settings, best_sharpe, epoch_sharpe)

    assert best_sharpe == 20

def test_training_fakedata():
    np.random.rand(1)
    n_time = 50
    n_markets = 3
    n_batch = 1
    n_sharpe = 10

    data1 = np.random.rand(n_time, 1) + 20

    data2 = np.random.rand(n_time, 1) + 5

    data3 = np.ones((n_time, 1)) * 1.01
    data3 = np.cumprod(data3, axis=0)
    data_all = np.hstack([data1, data2, data3] *4)
    data_in = np.hstack([data1, data2, data3])
    assert data_all.shape == (n_time, n_markets * 4)

    n_ftrs = data_in.shape[1]
    n_time = data_all.shape[0]-20
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': nn,
                'num_epochs': 20,
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
                'allow_shorting': True}

    
    settings = training(settings, all_data=data_in, market_data=data_all)
    nn_pos = settings['nn']._positions_np(batch_in=data_in[None, :n_time, :])
    sharpe = compute_numpy_sharpe(positions=nn_pos, prices=data_all[None, :n_time, :])

    assert sharpe > 1000

def temp_test_training_fakedata_2():
    np.random.rand(1)
    n_time = 50
    n_markets = 3
    n_batch = 1
    n_sharpe = 10

    data1 = np.random.rand(n_time, 1) + 20

    data2 = np.random.rand(n_time, 1) + 20

    data3 = np.random.rand(n_time, 1) + 20

    data_all = np.hstack([data1, data2, data3] *4)
    data_in = np.hstack([data1, data2, data3])
    assert data_all.shape == (n_time, n_markets * 4)
    n_ftrs = data_in.shape[1]
    n_time = data_all.shape[0] - 25
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    settings = {'nn': nn,
                'num_epochs': 100,
                'n_time': n_time,
                'n_sharpe': n_sharpe,
                'horizon': n_time - n_sharpe + 1,
                'val_period': 0,
                'lr': 1e-1,
                'lr_mult_base': 0.1,
                'batch_size': 25,
                'iter': 0,
                'lbd': 0,
                'realized_sharpe': [],
                'saved_val_sharpe': [],
                'realized_sharpe': [],
                'retrain_interval': np.nan,
                'allow_shorting': True}

    horizon = settings['horizon']
    # Predict prices at time = horizon+1
    nn_pos_b4 = np.vstack([settings['nn'].predict(data_in[i:horizon+i, :]) 
                          for i in range(data_in.shape[0]-horizon -1)])
    poscheck = nn_pos_b4[None, :, :]
    pricecheck = data_all[None, horizon+1:, :]


    sharpe_b4 = compute_numpy_sharpe(positions=poscheck, prices=pricecheck)
    print sharpe_b4
    settings = training(settings, all_data=data_in, market_data=data_all)
    settings['nn'].load()

    nn_pos_after = np.vstack([settings['nn'].predict(data_in[i:horizon+i, :]) 
                             for i in range(data_in.shape[0]-horizon -1)])

    nn_pos_after = np.vstack(nn_pos_after)
    sharpe_after = compute_numpy_sharpe(positions=nn_pos_after[None, :, :],
                                        prices=data_all[None, horizon:, :])
