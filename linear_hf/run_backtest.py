import random

import numpy as np
import joblib

import neuralnet
from preprocessing import non_nan_markets


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    all_data = np.array(OPEN)
    market_data = np.hstack([OPEN, CLOSE, HIGH, LOW])

    if settings['iter'] == 0:
        # Define a new neural net.
        settings['nn'] = neuralnet.Linear(n_ftrs=all_data.shape[1], 
                                          n_markets=OPEN.shape[1],
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'])

    # Train the neural net on current data.
    # for epoch_id in range(settings['num_epochs']):
    #     for batch_in, batch_out:
    #         settings['nn'].train_step(
    #             batch_in= , 
    #             batch_out=, 
    #             lr=settings['lr'])
    
    # Predict a portfolio.
    horizon = settings['n_time'] - settings['n_sharpe'] + 1
    positions = settings['nn'].predict(all_data[-horizon:])
    settings['iter'] += 1
    return np.ones(OPEN.shape[1]), settings


def mySettings():
    settings={}
    # Futures Contracts
    settings['n_time'] = 20 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 10 # This many timesteps to compute Sharpes.
    settings['lbd'] = .001 # L1 regularizer strength.
    settings['num_epochs'] = 1 # Number of epochs each day.
    settings['lr'] = 1e-4 # Learning rate.
    settings['iter'] = 0
    settings['lookback']=500
    settings['budget']=10**6
    settings['slippage']=0.05
    settings['beginInSample'] = '20090102'
    settings['endInSample'] = '20150101'
    settings['learn_causality'] = True

    # Only keep markets that have not died out by beginInSample.
    np.random.seed(1)
    random.seed(1)
    settings['markets']  = non_nan_markets(settings['beginInSample'], 
                                           settings['endInSample'], 
                                           lookback=settings['lookback'])
    settings['markets'] += ['CASH']
    settings['markets'] = settings['markets'][:13]
    print(settings['markets'])
    return settings



if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print(results['stats'])
