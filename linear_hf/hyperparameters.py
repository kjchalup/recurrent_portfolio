"""Determine the optimal hyperparameters for the neural net."""
import joblib
import itertools
import os

import random
import numpy as np

import quantiacsToolbox
from preprocessing import load_nyse_markets
from run_backtest import myTradingSystem

def powerset(iterable):
    """ Returns the set of all subsets of the iterable.

    Args:
      iterable: A Python iterable.

    Returns:
      result: The set of all subsets of the iterable, including the empty set.
    """
    all_vals = list(iterable)

    result = list(itertools.chain.from_iterable(
        itertools.combinations(all_vals, this_val)
        for this_val in range(len(all_vals)+1)))
    return result

# Define constants for use in choosing hyperparameters.
LBDS = 10.**np.arange(-5, 3) + [0.]
CHOICES = {'n_time': range(21, 50), # Timesteps in one datapoint.
           'lbd': LBDS,              # L1 regularizer strength.
           'num_epochs': [100],   # Number of epochs each day.
           'batch_size': [32, 64, 128],  # Batch size.
           'lr': 10.**np.arange(-7, -1),  # Learning rate.
           'allow_shorting': [False],
           'lookback' : [2000],
           'val_period' : [0],
           'val_sharpe_threshold' : [-np.inf, 0],
           'retrain_interval' : range(1, 101),
           'data_types' : [[1] + list(j) for j in powerset([4, 10, 12])],
           'cost_type': ['mixed_return'],
           'lr_mult_base': [1., .1, .01, .001],
           'restart_variables': [False]}#[True, False]}

N_SHARPE_MIN = 10 # Minimum value for n_sharpe.
N_SHARPE_GAP = 10 # n_sharpe's max is this much less than n_time.
N_RUNS = 1000 # ??? - KC

def mySettings(): # pylint: disable=invalid-name,too-many-arguments
    """ Settings for strategy. """
    settings = {}
    settings = joblib.load('saved_data/hypers.pkl')

    # Only keep markets that have not died out by beginInSample.
    random.seed(1)
    all_nyse = load_nyse_markets('20000601', None)
    np.random.seed(1)
    settings['markets'] = np.random.choice(all_nyse, 1000).tolist() + ['CASH']
    return settings

def supply_hypers():
    """Supply hyperparameters to optimize the neural net."""
    # Get random choices from the ranges (inclusive).
    settings = {}
    for setting in CHOICES:
        settings[setting] = random.choice(CHOICES[setting])

    # Get n_sharpe using n_time.
    settings['n_sharpe'] = random.randint(
        N_SHARPE_MIN, settings['n_time'] - N_SHARPE_GAP)

    return settings

if __name__ == '__main__':
    if os.path.isfile('saved_data/hyper_new_results_local.pkl'):
        HYPER_RESULTS = joblib.load('saved_data/hyper_new_results_local.pkl')
    else:
        HYPER_RESULTS = []

    # Get hyperparameters.
    SETTINGS = supply_hypers()

    # Other SETTINGS.
    SETTINGS['horizon'] = SETTINGS['n_time'] - SETTINGS['n_sharpe'] + 1
    if SETTINGS['cost_type'] != 'sharpe':
        SETTINGS['val_sharpe_threshold'] = -np.inf
    SETTINGS['iter'] = 0
    SETTINGS['budget'] = 10**6
    SETTINGS['slippage'] = 0.05
    SETTINGS['beginInSample'] = '20020102'
    SETTINGS['endInSample'] = '20131201'
    SETTINGS['realized_sharpe'] = []
    SETTINGS['saved_val_sharpe'] = []
    SETTINGS['val_sharpe'] = -np.inf
    SETTINGS['dont_trade'] = False

    # Save settings for use in test.
    joblib.dump(SETTINGS, 'saved_data/hypers.pkl')

    # Run the strategy.
    print [str(hyper) +': ' + str(SETTINGS[hyper])
           for hyper in SETTINGS and CHOICES]
    print ['n_time: ' + str(SETTINGS['n_time'])]
    try:
        RESULTS = quantiacsToolbox.runts(__file__, plotEquity=False)
        # Show the results.
        RESULTS['settings']['nn'] = None
        print RESULTS['stats']
    except ValueError:
        print 'Strategy failed so bad, we are skipping it.'
        RESULTS = SETTINGS

    # Reduce the size of the results files.
    HYPER_RESULTS.append(RESULTS)

    # Save the results
    joblib.dump(HYPER_RESULTS, 'saved_data/hyper_new_results_local.pkl')
