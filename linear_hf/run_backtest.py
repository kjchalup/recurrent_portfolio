import sys
import joblib
import random

from sklearn.preprocessing import StandardScaler
import numpy as np

import neuralnet
import chunknet
from preprocessing import non_nan_markets
from preprocessing import nan_markets
from preprocessing import returns_check

from preprocessing import load_nyse_markets
from preprocessing import preprocess
from batching_splitting import split_validation_training
from costs import compute_numpy_sharpe

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, CLOSE_LASTTRADE,
                    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, 
                    TOTALCAP, exposure, equity, settings, fundEquity):

    # Checks if we should end the backtest run.
    kill_backtest_run(fundEquity)

    # Preprocess the data
    market_data, all_data, should_retrain = preprocess(
        settings['markets'], OPEN, CLOSE, HIGH, LOW, VOL, DATE,
        CLOSE_LASTTRADE, CLOSE_ASK, CLOSE_BID, RETURN, SHARE,
        DIVIDEND, TOTALCAP, postipo = 100, filler = 0.123456789,
        data_types = settings['data_types'])
    
    # NEED TO INCLUDE THIS AS A HYPERPARAMETER!
    # Calculate Sharpe between training intervals

    recent_cost = calculate_recent(iteration=settings['iter'],
                                   retrain_interval=settings['retrain_interval'],
                                   exposure=exposure,
                                   market_data=market_data,
                                   cost=settings['cost_type'])
    print_things(settings['iter'], DATE[-1], 
                 fundEquity[-1], settings['best_val_sharpe'], recent_cost)

    # Initialize neural net.
    if settings['iter'] == 0:
        settings = init_nn(settings, all_data.shape[1], 'linear')

    # Train the neural net on current data.
    if settings['iter'] % settings['retrain_interval'] == 0:
        if settings['restart_variables']:
            settings['nn'].restart_variables()

        # Train the neural net for settings['num_epoch'] times
        settings = training(settings=settings, 
                            all_data=all_data, market_data=market_data)

        # After all epochs, check if the best_sharpe_val allows for trading.
        if settings['val_period'] > 0:
            if settings['best_val_sharpe'] > settings['val_sharpe_threshold']:
                settings['dont_trade'] = False
            else:
                settings['dont_trade'] = True

    # Save validation sharpes and actualized sharpes!
    settings['realized_sharpe'].append(recent_cost)

    if settings['val_period'] == 0:
        settings['saved_val_sharpe'].append(np.nan)
    else:
        settings['saved_val_sharpe'].append(settings['best_val_sharpe'])
    settings['iter'] += 1

    # Predict a portfolio.
    settings['nn'].load()
    positions = settings['nn'].predict(all_data[-settings['horizon']:])

    # Set positions to zero, cash to 1 if don't trade is True.
    if settings['dont_trade']:
        positions = dont_trade_positions(positions, settings)
    return positions, settings

def mySettings():
    settings = {}
    # Futures Contracts
    settings['n_time'] = 400 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 252 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = 1. # L1 regularizer strength.
    settings['num_epochs'] = 30 # Number of epochs each day.
    settings['batch_size'] = 128
    settings['val_period'] = 1
    settings['lr'] = 1e-7 # Learning rate.
    settings['dont_trade'] = False # If on, don't trade.
    settings['iter'] = 0
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    #settings['beginInSample'] = '20090102'
    #settings['endInSample'] = '20131231'
    settings['beginInSample'] = '20000601'
    settings['endInSample'] = '20140101'
    settings['val_sharpe_threshold'] = -np.inf
    settings['retrain_interval'] = 252
    settings['realized_sharpe'] = []
    settings['saved_val_sharpe'] = []
    settings['best_val_sharpe'] = -np.inf
    settings['cost_type'] = 'sharpe'
    settings['allow_shorting'] = False
    settings['lr_mult_base'] = 1.
    settings['restart_variables'] = True
    ''' Pick data types to feed into neural net. If empty, only CLOSE will be used. 
    Circle dates added automatically if any setting is provided.
    0 = OPEN
    1 = CLOSE
    2 = HIGH
    3 = LOW
    4 = VOL
    5 = CLOSE_LASTTRADE
    6 = CLOSE_ASK
    7 = CLOSE_BID
    8 = RETURNS
    9 = SHARES
    10 = DIVIDENDS
    11 = TOTALCAPS
    12 = DATE
    '''
    settings['data_types'] = [0]

    settings['markets'] = load_nyse_markets(start_date=settings['beginInSample'],
                                            end_date=settings['endInSample'],
                                            lookback=0,
                                            postipo=0)
    settings['markets'] = settings['markets'][-20:] + ['CASH']
    print settings['markets']
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print results['stats']


def calculate_recent(iteration, retrain_interval, exposure, market_data, cost='sharpe'):
    """ Calculate the realized sharpe ratios from the output of the neural net
        using the latest trading positions.

    Args:
        iteration: iteration number in backtester, used to calculate n_days_back
        retrain_interval: how often the nn retrains
        exposure: output from the backtester, settings['exposure'] (ntimesteps, nmarkets)
        market_data: output from the backtester, [open close high low] (ntimesteps, nmarkets)

    Returns:
        Cost function estimate:
            if cost = 'sharpe', returns annualized sharpe since last retrain
            if cost = 'min_return', returns minimium of the daily return since last retrain
            if cost = 'mean_return', returns the mean of the returns since last retrain
            if cost = 'mixed_return', returns mean*min of returns since last retrain
    """

    # Calculate Sharpe between training intervals. iteration-1 is used because we score one day back.
    n_days_back = np.mod(iteration,retrain_interval)
     
    # Only start scoring realized sharpes from greater than 2!
    if n_days_back > 2:
        recent_sharpe = compute_numpy_sharpe(positions=exposure[None, -n_days_back-3:-1, :],
                                             prices=market_data[None, -n_days_back-2:, :],
                                             slippage=0.05, n_ignore=2)
        recent_returns = compute_numpy_sharpe(positions=exposure[None, -n_days_back-3:-1, :],
                                              prices=market_data[None, -n_days_back-2:, :],
                                              slippage=0.05, n_ignore=2, return_returns=True)

        if np.isnan(recent_sharpe):
            # NaNs out when all positions are cash, therefore std.dev(ret) = 0
            recent_sharpe = 0
    else:
        # Return nans for the first couple days.
        recent_sharpe = np.nan
        recent_returns = np.array([np.nan, np.nan])

    if cost == 'sharpe':
        return recent_sharpe
    elif cost == 'min_return':
        return recent_returns.min()
    elif cost == 'mean_return':
        return recent_returns.mean()
    elif cost == 'mixed_return':
        return recent_returns.mean() + recent_returns.min()

def print_things(iteration, DATE, fundEquity, val_sharpe, recent_cost):
    """ Prints out things

    Args:
        iteration: iteration in backtester
        DATE: last date
        fundEquity: last fundEquity amount
        val_sharpe: most recent validation sharpe saved in settings
        recent_cost: most recent recent_cost.
    """
    print('Iter {} [{}], fundEquity {}.'.format(iteration,
                                                DATE,
                                                fundEquity))
    if iteration > 1:
        print('[Recent validation costfn] Recent costfn: [{}] {}'.format(val_sharpe,
                                                                         recent_cost))


def dont_trade_positions(positions, settings):
    """ Sets positions to zero, and cash to 1

    Args:
        positions: positions output from neuralnet
        settings: needed to find 'CASH' index
    Returns:
        positions: all zeroed out except for CASH, which is 1
    """

    positions *= 0
    cash_index = settings['markets'].index('CASH')
    positions[cash_index] = 1
    return positions


def calc_batches(n_timesteps, settings):
    """ Calculates the total groups of batch_size that are one epoch.
    
    Args:
        n_timesteps: total number of timesteps
        settings: takes horizon, val_period, n_sharpe, and batch_size
    Returns:
        batches_per_epoch: the floor of total_possible_timesteps/batch_size,
                           where total_possible is taken from batches
    """
    if settings['val_period'] > 0:
        batches_per_epoch = int(np.floor((n_timesteps -
                                        settings['horizon'] -
                                        settings['val_period'] -
                                        2 * settings['n_sharpe'] + 1)
                                        / float(settings['batch_size'])))
    else:
        batches_per_epoch = int(np.floor((n_timesteps -
                                          settings['horizon'] -
                                          settings['n_sharpe'] + 1)
                                          / float(settings['batch_size'])))
    return batches_per_epoch


def lr_calc(settings, epoch_id):
    """ Learning rate update based on epoch_id

    Args:
        settings: used to caluclate an lr multiplier
        epoch_id: used to calculate the new lr
    Return:
        lr_new: new learning rate, dependent on epoch_id, lr, lr_mult_base.
    """
    lr_mult = settings['lr_mult_base'] ** (1. / settings['num_epochs'])
    lr_new = settings['lr'] * lr_mult ** epoch_id

    return lr_new

def loss_calc(settings, all_batch, market_batch):
    """ Calculates loss from neuralnet

    Args:
        settings: contains the neural net
        all_batch: the inputs to neural net
        market_batch: [open close high low] used to calculate loss
    Returns:
        cost: loss - l1 penalty
    """
    loss = settings['nn'].loss_np(all_batch, market_batch)
    l1_loss = settings['nn'].l1_penalty_np()
    return -(loss - l1_loss)

def update_nn(settings, best_sharpe, epoch_sharpe):
    """ Saves neural net and updates best_sharpe if better in this epoch.

    Args:
        settings: contains the neuralnet
        best_sharpe: the previously highest sharpe (or cost function)
        epoch_sharpe: the sharpe for the current epoch (either validation or avg)

    Returns:
        settings: saved neuralnet in settings
        best_sharpe: updated new sharpe or old best sharpe
    """
    if epoch_sharpe > best_sharpe:
        best_sharpe = epoch_sharpe
        settings['nn'].save()
        settings['best_val_sharpe'] = epoch_sharpe

    return settings, best_sharpe


def init_nn(settings, n_ftrs, nn_type):
    """ Intializes the neural net

    Args:
        settings: where the neuralnet gets initialized
        n_ftrs: size of the neuralnet input layer
        nn_type: type of neural net
    Returns:
        settings: a dict with ['nn'] which is the initialized neuralnet.
    """
    print 'Initializing net...\n'
    if nn_type == 'linear':
        settings['nn'] = neuralnet.Linear(n_ftrs=n_ftrs,
                                          n_markets=len(settings['markets']),
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'],
                                          allow_shorting=settings['allow_shorting'],
                                          cost=settings['cost_type'])
    print 'Done with initializing neural net!'
    return settings

def kill_backtest_run(fundEquity):
    """ Raises an exception to kill the backtest."""
    if fundEquity[-1] < .75:
        raise ValueError('Strategy lost too much money')

def training(settings, all_data, market_data):
    """ Trains the neuralnet.
    Total steps:
    1) train for settings['num_epochs']
        a) calculates new learning rate for each epoch
    2) saves neural net if the epoch has a better val_sharpe or tr_sharpe
    3) saves the best_val_sharpe to settings['best_val_sharpe']

    Args:
        settings: contains nn to be trained, as well as other settings
        all_data: total data fed into neuralnet (ntimesteps, nftrs)
        market_data: data to score neuralnet (ntimesteps, nmarkets*4)
    Returns:
        settings: updated neural net, and best_val_sharpe
    """
    # Intiailize sharpe ratios
    best_val_sharpe = -np.inf
    best_tr_sharpe = -np.inf
    batches_per_epoch = calc_batches(all_data.shape[0], settings)
    # Start an epoch!
    for epoch_id in range(settings['num_epochs']):
        seed = np.random.randint(10000)
        tr_sharpe = 0.
        val_sharpe = 0.
        lr_new = lr_calc(settings, epoch_id)
        # Train an epoch.
        for batch_id in range(batches_per_epoch):
            # Split data into validation and training batches.
            all_val, market_val, all_batch, market_batch = split_validation_training(
                all_data=all_data, market_data=market_data,
                valid_period=settings['val_period'],
                horizon=settings['horizon'],
                n_for_sharpe=settings['n_sharpe'],
                batch_id=batch_id,
                batch_size=settings['batch_size'],
                randseed=seed)
            # Train.
            settings['nn'].train_step(batch_in=all_batch, 
                                      batch_out=market_batch, lr=lr_new)
            tr_sharpe += loss_calc(settings, all_batch, market_batch)


        # Calculate sharpes for the epoch
        tr_sharpe /= batches_per_epoch
        if settings['val_period'] > 0:
            val_sharpe = loss_calc(settings, all_batch=all_val, market_batch=market_val)
        # Update neural net, and attendant values if NN is better than previous.
        if settings['val_period'] > 0:
            settings, best_val_sharpe = update_nn(settings, best_val_sharpe, val_sharpe)
        else:
            settings, best_tr_sharpe = update_nn(settings, best_tr_sharpe, tr_sharpe)

        # Write out data for epoch.
        sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
            epoch_id, val_sharpe, tr_sharpe))
        sys.stdout.flush()

    return settings
