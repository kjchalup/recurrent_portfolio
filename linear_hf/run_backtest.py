import random

import numpy as np
import joblib

import neuralnet
from preprocessing import non_nan_markets
from batching_splitting import split_validation_training


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    all_data = np.array(OPEN)
    market_data = np.hstack([OPEN, CLOSE, HIGH, LOW])
    print('Iter {} [{}], equity {}.'.format(settings['iter'], 
                                            DATE[-1],
                                            fundEquity[-1]))

    if settings['iter'] == 0:
        # Define a new neural net.
        settings['nn'] = neuralnet.Linear(n_ftrs=all_data.shape[1], 
                                          n_markets=OPEN.shape[1],
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'])

    # Train the neural net on current data.
    batches_per_epoch = int(np.floor((all_data.shape[0] -
                                      settings['horizon'] -
                                      2 * settings['n_sharpe'] + 2)
                                     /float(settings['batch_size'])))
    for epoch_id in range(settings['num_epochs']):
        for batch_id in range(batches_per_epoch):
            _, _, all_batch, market_batch = split_validation_training(
                all_data, market_data, valid_period=0, 
                horizon=settings['horizon'], 
                n_for_sharpe=settings['n_sharpe'],
                batch_id=batch_id, 
                batch_size=settings['batch_size'],
                randseed=epoch_id)

            settings['nn'].train_step(
                batch_in=all_batch, 
                batch_out=market_batch, 
                lr=settings['lr'])
    
    # Predict a portfolio.
    positions = settings['nn'].predict(all_data[-settings['horizon']:])
    settings['iter'] += 1
    return np.ones(OPEN.shape[1]), settings


def mySettings():
    settings={}
    # Futures Contracts
    settings['n_time'] = 20 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 10 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = .001 # L1 regularizer strength.
    settings['num_epochs'] = 1 # Number of epochs each day.
    settings['batch_size'] = 32
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
