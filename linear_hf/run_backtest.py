import sys
import random

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

import neuralnet
import chunknet
from preprocessing import non_nan_markets
from preprocessing import nan_markets
from preprocessing import returns_check
from preprocessing import load_nyse_markets
from preprocessing import preprocess
from batching_splitting import split_validation_training
from costs import compute_numpy_sharpe

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    

    market_data, all_data, should_retrain = preprocess(
        settings['markets'], OPEN, CLOSE, HIGH, LOW, VOL, DATE, 
        CLOSE_LASTTRADE, CLOSE_ASK, CLOSE_BID, RETURN, SHARE, 
        DIVIDEND, TOTALCAP, postipo=100, filler=0.123456789, 
        data_types = settings['data_types'])

    all_data = StandardScaler().fit_transform(all_data)
    # Calculate Sharpe between training intervals
    n_days_back = np.mod(settings['iter']-1,settings['retrain_interval'])
    
    if n_days_back > 2:
        recent_sharpe=compute_numpy_sharpe(positions=exposure[None, -n_days_back-4:-1, :],
                             prices=market_data[None, -n_days_back-3:, :],
                             slippage=0.05,
                             n_ignore=2)
        if np.isnan(recent_sharpe):
            # NaNs out when all positions are cash, therefore std.dev(ret) = 0
            recent_sharpe = 0
    else:
        recent_sharpe = np.nan
    
    print('Iter {} [{}], equity {}.'.format(settings['iter'], 
                                            DATE[-1],
                                            fundEquity[-1]))
    if fundEquity[-1] < .75:
        settings['nn'].sess.close()
        raise ValueError('Strategy lost too much money')

    if settings['iter'] > 2:
        print('[Recent validation sharpe] Recent sharpe: [{}] {}'.format(
                                            settings['val_sharpe'],
                                            recent_sharpe))
    if settings['iter'] == 0:
        print 'Initializing net...\n'
        # Define a new neural net.
        settings['nn'] = chunknet.ChunkLinear(n_ftrs=all_data.shape[1], 
                                              n_markets=OPEN.shape[1],
                                              n_time=settings['n_time'],
                                              n_sharpe=settings['n_sharpe'],
                                              n_chunks=10,
                                              lbd=settings['lbd'],
                                              allow_shorting=settings['allow_shorting'],
                                          cost=settings['cost_type'])
        print 'Done with initializing neural net!'

    # Train the neural net on current data.
    best_val_sharpe = -np.inf
    if settings['iter'] % settings['retrain_interval'] == 0:
        best_val_sharpe = -np.inf
        best_tr_loss = np.inf
        if settings['restart_variables']:
            settings['nn'].restart_variables()
        lr_mult = settings['lr_mult_base'] ** (1. / settings['num_epochs'])
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
                (all_val, market_val, 
                 all_batch, market_batch) = split_validation_training(
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
                    lr = settings['lr'] * lr_mult ** epoch_id)
                loss = settings['nn'].loss_np(all_batch, market_batch)
                l1_loss = settings['nn'].l1_penalty_np()
                tr_sharpe += -(loss - l1_loss)

            if settings['val_period'] > 0:
                val_loss = settings['nn'].loss_np(all_val, market_val)
                val_l1_loss = settings['nn'].l1_penalty_np()
                val_sharpe = -(val_loss - val_l1_loss)
                
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    settings['nn'].save()

                if best_val_sharpe > settings['val_sharpe_threshold']:
                    settings['dont_trade'] = False
                else:
                    settings['dont_trade'] = True
                
                # Record val_sharpe for results
                settings['val_sharpe'] = best_val_sharpe

            elif loss < best_tr_loss:
                best_tr_loss = loss
                settings['nn'].save()

            tr_sharpe /= batches_per_epoch
            sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
                epoch_id, val_sharpe, tr_sharpe))
            sys.stdout.flush()
        
    # Predict a portfolio.
    settings['nn'].load()
    positions = settings['nn'].predict(all_data[-settings['horizon']:])
    if settings['dont_trade']:
        positions *= 0 
        cash_index = settings['markets'].index('CASH')
        positions[cash_index] = 1
    
    # Save validation sharpes and actualized sharpes!
    settings['realized_sharpe'].append(recent_sharpe)
    settings['saved_val_sharpe'].append(best_val_sharpe)
    
    settings['iter'] += 1
    return positions, settings


def mySettings():
    settings={}
    # Futures Contracts
    settings['n_time'] =  31 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 14 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = 1. # L1 regularizer strength.
    settings['num_epochs'] = 48 # Number of epochs each day.
    settings['batch_size'] = 128
    settings['val_period'] = 32
    settings['lr'] = 1e-9 # Learning rate.
    settings['dont_trade'] = False # If on, don't trade.
    settings['iter'] = 0
    settings['lookback'] = 1800
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    #settings['beginInSample'] = '20090102'
    #settings['endInSample'] = '20131231'
    settings['beginInSample'] = '20000601'
    settings['endInSample'] = '20140101'
    settings['val_sharpe_threshold'] = -np.inf
    settings['retrain_interval'] = 51
    settings['realized_sharpe'] = []
    settings['saved_val_sharpe'] = []
    settings['val_sharpe'] = -np.inf
    settings['cost_type'] = 'sharpe'
    settings['allow_shorting'] = True
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
    settings['data_types'] = [1]
    
    # Hard code markets for testing.
    settings['markets'] = load_nyse_markets(start_date=settings['beginInSample'],
                                            end_date=settings['endInSample'],
                                            lookback=0,
                                            postipo=0)
    # Set the n_markets to be a multiple of 10 (including CASH) so
    # we can chunk!
    n_markets = ((len(settings['markets']) + 1) / 10) * 10 - 1
    settings['markets'] = settings['markets'][:n_markets] + ['CASH']
    print(settings['markets'])
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print(results['stats'])
