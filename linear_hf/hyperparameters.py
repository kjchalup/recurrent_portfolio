"""Determine the optimal hyperparameters for the neural net."""
import sys
import random

import numpy as np
import joblib

import neuralnet as neuralnet   
from preprocessing import non_nan_markets
from batching_splitting import split_validation_training

# Define constants for use in choosing hyperparameters.
LBDS = np.append(10.**np.arange(-10, 2), [0.])
CHOICES = {'n_time': range(20, 253), # Timesteps in one datapoint.
           'lbd': LBDS,              # L1 regularizer strength.
           'num_epochs': range(1, 51),   # Number of epochs each day.
           'batch_size': [32, 64, 128],  # Batch size.
           'lr': 10.**np.arange(-7, 0),  # Learning rate.
           'allow_shorting': [True, False],}
N_SHARPE_MIN = 10               # Minimum value for n_sharpe.
N_SHARPE_GAP = 10               # n_sharpe's max is this much less than n_time.
N_RUNS = 2


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, settings, fundEquity): # pylint: disable=invalid-name,too-many-arguments
    """Neural net trading system."""
    all_data = np.array(OPEN)
    market_data = np.hstack([OPEN, CLOSE, HIGH, LOW])
    print('Iter {} [{}], equity {}.'.format(settings['iter'],
                                            DATE[-1],
                                            fundEquity[-1]))

    if settings['iter'] == 0:
        # Define a new neural net.
        settings['nn'] = neuralnet.Linear(
            n_ftrs=all_data.shape[1],
            n_markets=OPEN.shape[1],
            n_time=settings['n_time'],
            n_sharpe=settings['n_sharpe'],
            lbd=settings['lbd'],
            allow_shorting=settings['allow_shorting'],)

    # Train the neural net on current data.
    if settings['iter'] % settings['n_sharpe'] == 0:
        settings['nn'].restart_variables()
        lr_mult = 1 #.01 ** (1. / settings['num_epochs'])
        batches_per_epoch = int(np.floor((all_data.shape[0] -
                                          settings['horizon'] -
                                          settings['val_period'] -
                                          2 * settings['n_sharpe'] + 1)
                                         /float(settings['batch_size'])))
        for epoch_id in range(settings['num_epochs']):
            seed = np.random.randint(10000)
            tr_sharpe = 0.
            val_sharpe = 0.
            for batch_id in range(batches_per_epoch):
                all_val, market_val, all_batch, market_batch = (
                    split_validation_training(
                        all_data, market_data,
                        valid_period=settings['val_period'],
                        horizon=settings['horizon'],
                        n_for_sharpe=settings['n_sharpe'],
                        batch_id=batch_id,
                        batch_size=settings['batch_size'],
                        randseed=seed)
                )
                settings['nn'].train_step(
                    batch_in=all_batch,
                    batch_out=market_batch,
                    lr=settings['lr'] * lr_mult ** epoch_id)
                loss = settings['nn'].loss_np(all_batch, market_batch)
                l1_loss = settings['nn'].l1_penalty_np()
                tr_sharpe += -(loss - l1_loss)
            if settings['val_period'] > 0:
                val_loss = settings['nn'].loss_np(all_val, market_val)
                val_l1_loss = settings['nn'].l1_penalty_np()
                val_sharpe = -(val_loss - val_l1_loss)
            tr_sharpe /= batches_per_epoch
            sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
                epoch_id, val_sharpe, tr_sharpe))
            sys.stdout.flush()

    # Predict a portfolio.
    positions = settings['nn'].predict(all_data[-settings['horizon']:])
    settings['iter'] += 1
    return positions, settings

def mySettings():
    """Settings for strategy."""
    settings = {}
    settings = joblib.load('saved_data/hypers.pkl')

    # Only keep markets that have not died out by beginInSample.
    np.random.seed(1)
    random.seed(1)
    settings['markets'] = non_nan_markets(settings['beginInSample'],
                                          settings['endInSample'],
                                          lookback=settings['lookback'])
    settings['markets'] = settings['markets'][:10]
    print(settings['markets'])
    return settings

def supply_hypers():
    """Supply hyperparameters to optimize the neural net."""

    np.random.seed()
    random.seed()

    # Get random choices from the ranges (inclusive).
    settings = {}
    for setting in CHOICES:
        settings[setting] = random.choice(CHOICES[setting])

    # Get n_sharpe using n_time.
    settings['n_sharpe'] = random.randint(N_SHARPE_MIN,
                                          settings['n_time'] - N_SHARPE_GAP)

    return settings

if __name__ == '__main__':
    HYPER_RESULTS = []
    for run in range(N_RUNS):
        # Get hyperparameters.
        SETTINGS = supply_hypers()

        # Other SETTINGS.
        SETTINGS['horizon'] = SETTINGS['n_time'] - SETTINGS['n_sharpe'] + 1
        SETTINGS['iter'] = 0
        SETTINGS['lookback'] = 1000
        SETTINGS['budget'] = 10**6
        SETTINGS['slippage'] = 0.05
        SETTINGS['val_period'] = 0
        SETTINGS['beginInSample'] = '20080102'
        SETTINGS['endInSample'] = '20131201'

        # Save settings for use in test.
        joblib.dump(SETTINGS, 'saved_data/hypers.pkl')

        # Run the strategy.
        import quantiacsToolbox
        RESULTS = quantiacsToolbox.runts(__file__, plotEquity=False)

        # Show the results.
        RESULTS['settings']['nn'] = None
        print([str(hyper) +': ' + str(SETTINGS[hyper])
               for hyper in SETTINGS and CHOICES])
        print(['n_time: ' + str(SETTINGS['n_time'])])
        print(RESULTS['stats'])

        # Reduce the size of the results files.
        # RESULTS['fundDate'] = None
        # RESULTS['marketEquity'] = None
        # RESULTS['returns'] = None
        # RESULTS['marketExposure'] = None
        HYPER_RESULTS.append(RESULTS)

    # Save the resutls
    joblib.dump(HYPER_RESULTS, 'saved_data/hyper_results.pkl')
