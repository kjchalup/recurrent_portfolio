""" Check to see if the same hyperparameters are supplied."""
import sys

import numpy as np

from linear_hf.choose_100_stocks import choose_100_stocks

# Define constants for use in choosing hyperparameters.
LBDS = 10.**np.arange(-5, 3) + [0.]
CHOICES = {'n_time': range(30, 100), # Timesteps in one datapoint.
           'lbd': LBDS,              # L1 regularizer strength.
           'num_epochs': [1, 5, 10, 50, 100],   # Number of epochs each day.
           'batch_size': [16, 32, 64, 128],  # Batch size.
           'lr': 10.**np.arange(-7, -1),  # Learning rate.
           'allow_shorting': [True, False],
           'lookback' : [1000, 800, 600, 400],
           'val_period' : [0, 0, 0, 0, 4, 8, 16],
           'val_sharpe_threshold' : [-np.inf, 0],
           'retrain_interval' : range(10, 252),
           'cost_type': ['sharpe', 'sortino', 'equality_sharpe',
                         'equality_sortino', 'min_return',
                         'mixed_return', 'mean_return'],
           'lr_mult_base': [1., .1, .01, .001],
           'causal_interval': [50, 100, 252, 252*2, 252*3, 252*4],
           'restart_variables': [True, False]}

N_SHARPE_MIN = 10 # Minimum value for n_sharpe.
N_SHARPE_GAP = 10 # n_sharpe's max is this much less than n_time.
N_RUNS = 1000

def supply_hypers(seed):
    """Supply hyperparameters to optimize the neural net."""
    # Get random choices from the ranges (inclusive).
    settings = {}
    for setting in CHOICES:
        np.random.seed(seed)
        settings[setting] = np.random.choice(CHOICES[setting])

    # Get n_sharpe using n_time.
    settings['n_sharpe'] = np.random.randint(
        N_SHARPE_MIN, settings['n_time'] - N_SHARPE_GAP)

    return settings

if __name__ == '__main__':
    # Choose SEED for random numbers based on count in the script.
    SEED = int(sys.argv[1])
    np.random.seed(SEED)

    # Get hyperparameters.
    SETTINGS = supply_hypers(SEED)

    # Use choose_100_stocks to supply markets.
    SETTINGS['markets'] = choose_100_stocks(SEED)

    print("n_time: " + str(SETTINGS['n_time']))
    print("n_sharpe: " + str(SETTINGS['n_sharpe']))
