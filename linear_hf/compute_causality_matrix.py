import sys
import random

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

from causality import causal_matrix
from preprocessing import non_nan_markets
from preprocessing import nan_markets

def myTradingSystem(OPEN, exposure, equity, settings, fundEquity):
    cm = causal_matrix(OPEN, verbose=False, method='nearest', n_neighbors=30, nruns=30)
    joblib.dump([settings, cm], 'saved_data/causality_matrix.pkl')
    return None

def mySettings():
    settings={}
    # Futures Contracts
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20090102'
    settings['endInSample'] = '20140101'

    # Only keep markets that have not died out by beginInSample.
    np.random.seed(1)
    random.seed(1)
    settings['markets']  = non_nan_markets(settings['beginInSample'], 
                                           settings['endInSample'], 
                                           lookback=settings['lookback'])
    #settings['markets'] = nan_markets(settings['beginInSample'],
    #                                  settings['endInSample'],
    #                                  lookback=settings['lookback'])
    settings['markets'] = settings['markets'][:10]
    print(settings['markets'])
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print(results['stats'])
