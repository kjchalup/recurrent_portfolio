"""Determine the optimal hyperparameters for the neural net."""
import random
import itertools
import os
import sys
import joblib

import numpy as np

import quantiacsToolbox

from linear_hf.preprocessing import load_nyse_markets
from linear_hf.run_backtest import myTradingSystem

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
CHOICES = {'n_time': range(30, 200), # Timesteps in one datapoint.
           'lbd': LBDS,              # L1 regularizer strength.
           'num_epochs': [1, 5, 10, 50, 100],   # Number of epochs each day.
           'batch_size': [16, 32, 64, 128],  # Batch size.
           'lr': 10.**np.arange(-7, -1),  # Learning rate.
           'allow_shorting': [True, False],
           'lookback' : [1000, 800, 600, 400],
           'val_period' : [0, 0, 0, 0, 4, 8, 16],
           'val_sharpe_threshold' : [-np.inf, 0],
           'retrain_interval' : range(10, 252),
           'data_types' : [[1], [1, 4], [1, 10], [1, 12]],
           'cost_type': ['sharpe, sortino, equality_sharpe, equality_sortino', 'min_return', 'mixed_return', 'mean_return'],
           'lr_mult_base': [1., .1, .01, .001],
           'causal_interval': [0],
           'restart_variables': [True, False]}

N_SHARPE_MIN = 10 # Minimum value for n_sharpe.
N_SHARPE_GAP = 10 # n_sharpe's max is this much less than n_time.
N_RUNS = 1000

def mySettings(): # pylint: disable=invalid-name,too-many-arguments
    """ Settings for strategy. """
    settings = {}
    settings = joblib.load('saved_data/hypers.pkl')
    '''
    # Only keep markets that have not died out by beginInSample.
    random.seed(1)
    all_nyse = load_nyse_markets(start_date='20000104', 
                                 end_date='20131231', postipo=0,
                                 lookback=0)
    settings['markets'] = all_nyse[:2699] + ['CASH']
    '''
    settings['markets'] = joblib.load('linear_hf/1000_stock_names.pkl')
    return settings

def supply_hypers():
    """Supply hyperparameters to optimize the neural net."""
    # Get random choices from the ranges (inclusive).
    settings = {}
    for setting in CHOICES:
        settings[setting] = np.random.choice(CHOICES[setting])

    # Get n_sharpe using n_time.
    settings['n_sharpe'] = np.random.randint(
        N_SHARPE_MIN, settings['n_time'] - N_SHARPE_GAP)

    return settings

if __name__ == '__main__':
    np.random.seed(int(sys.argv[1]))
    results_fname = 'saved_data/hyper_100_noncsl_results.pkl'
    if os.path.isfile(results_fname):
        HYPER_RESULTS = joblib.load(results_fname)
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
    SETTINGS['beginInSample'] = '20000104'
    SETTINGS['endInSample'] = '20131201'
    SETTINGS['realized_sharpe'] = []
    SETTINGS['saved_val_sharpe'] = []
    SETTINGS['best_val_sharpe'] = []
    SETTINGS['val_sharpe'] = -np.inf
    SETTINGS['dont_trade'] = False
    SETTINGS['n_chunks'] = 1
    SETTINGS['nn_type'] = 'linear'
    SETTINGS['causal_matrix'] = None
    # Save settings for use in test.
    joblib.dump(SETTINGS, 'saved_data/hypers.pkl')

    # Run the strategy.
    print [str(hyper) +': ' + str(SETTINGS[hyper])
           for hyper in SETTINGS and CHOICES]
    print ['n_time: ' + str(SETTINGS['n_time'])]
    try:
        RESULTS = quantiacsToolbox.runts(
            __file__, plotEquity=False, fname='linear_hf/1000_nyse_stocks.pkl')[
                np.random.choice(1000, 99, replace=False)] + ['CASH']

        # Show the results.
        RESULTS['settings']['nn'] = None
        print RESULTS['stats']
    except ValueError:
        print 'Strategy failed so bad, we are skipping it.'
        RESULTS = SETTINGS

    # Reduce the size of the results files.
    HYPER_RESULTS.append(RESULTS)

    # Save the results
    joblib.dump(HYPER_RESULTS, results_fname)
