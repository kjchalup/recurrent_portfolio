import sys
import random

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

import neuralnet
from preprocessing import non_nan_markets
from preprocessing import nan_markets
from preprocessing import returns_check
from preprocessing import preprocess
from batching_splitting import split_validation_training

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    
    n_markets=len(settings['markets'])
    # Returns check to make sure nothing crazy happens!
    returns_check(OPEN, CLOSE, HIGH, LOW, DATE, settings['markets'])

    market_data, all_data, should_retrain = preprocess(
                                                        settings['markets'], OPEN, CLOSE, HIGH, LOW,
                                                        VOL, DATE, CLOSE_LASTTRADE, CLOSE_ASK, 
                                                        CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP,
                                                        postipo=100, filler=0.123456789)
    # Returns check afte preprocessing to make sure nothing crazy happens!
    returns_check(market_data[:,:n_markets],
                    market_data[:,n_markets:n_markets*2],
                    market_data[:,n_markets*2:n_markets*3],
                    market_data[:,n_markets*3:n_markets*4],
                    DATE, settings['markets'])
    # Run backtester without preprocessing
    #market_data = np.hstack([OPEN, CLOSE, HIGH, LOW])
    #all_data = StandardScaler().fit_transform(OPEN)#np.hstack([OPEN, VOL, DIVIDEND, TOTALCAP]))
    
    # Run backtester with preprocessing
    if settings['data_types'] is None:
        all_data = Standardscaler().fit_transform(all_data[:,:n_markets])
    else:
        data = []
        for j in settings['data_types']:
            data.append(all_data[:,n_markets*(j):n_markets*(j+1)])
    
    # Stacks chosen data back into correct shape!
    all_data = np.hstack(data)
    
    print('Iter {} [{}], equity {}.'.format(settings['iter'], 
                                            DATE[-1],
                                            fundEquity[-2]))
    #import pdb;pdb.set_trace()
    if settings['iter'] == 0:
        print 'Initializing net...\n'
        # Define a new neural net.
        settings['nn'] = neuralnet.Linear(n_ftrs=all_data.shape[1], 
                                          n_markets=OPEN.shape[1],
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'],
                                          allow_shorting=False)
        print 'Done with initializing neural net!'
    # Train the neural net on current data.
    if settings['iter'] % settings['n_sharpe'] == 0:
        settings['nn'].restart_variables()
        lr_mult = .01 ** (1. / settings['num_epochs'])
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
                all_val, market_val, all_batch, market_batch = split_validation_training(
                    all_data, market_data, 
                    valid_period=settings['val_period'], 
                    horizon=settings['horizon'], 
                    n_for_sharpe=settings['n_sharpe'],
                    batch_id=batch_id, 
                    batch_size=settings['batch_size'],
                    randseed=seed)

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
                #import pdb;pdb.set_trace()
                if val_sharpe > 1:
                    settings['dont_trade'] = False
                else:
                    settings['dont_trade'] = True

            tr_sharpe /= batches_per_epoch
            sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
                epoch_id, val_sharpe, tr_sharpe))
            sys.stdout.flush()

    # Predict a portfolio.
    positions = settings['nn'].predict(all_data[-settings['horizon']:])
    if settings['dont_trade']:
        positions *= np.nan
    settings['iter'] += 1
    return positions, settings


def mySettings():
    settings={}
    # Futures Contracts
    settings['n_time'] =  160 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 100 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = .1 # L1 regularizer strength.
    settings['num_epochs'] = 10 # Number of epochs each day.
    settings['batch_size'] = 128
    settings['val_period'] = 1
    settings['lr'] = 1e-3 # Learning rate.
    settings['dont_trade'] = False # If on, don't trade.
    settings['iter'] = 0
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    #settings['beginInSample'] = '20090102'
    #settings['endInSample'] = '20131231'
    settings['beginInSample'] = '20090102'
    settings['endInSample'] = '20140101'

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
    '''
    settings['data_types'] = [0,1]
    settings['data_types'] = np.sort(settings['data_types'])
    
    
    # Only keep markets that have not died out by beginInSample.
    np.random.seed(1)
    random.seed(1)
    settings['markets']  = non_nan_markets(settings['beginInSample'], 
                                           settings['endInSample'], 
                                           lookback=settings['lookback'])
    #settings['markets'] = nan_markets(settings['beginInSample'],
    #                                  settings['endInSample'],
    #                                  lookback=settings['lookback'])
    settings['markets'] = settings['markets'][:10] + ['CASH']
    print(settings['markets'])
    return settings



if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print(results['stats'])
