import numpy as np
import joblib

from linear_hf.causality import causal_matrix_ratios
from linear_hf.preprocessing import preprocess

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    market_data, all_data, should_retrain = preprocess(
        settings['markets'], OPEN, CLOSE, HIGH, LOW, VOL, DATE,
        CLOSE_LASTTRADE, CLOSE_ASK, CLOSE_BID, RETURN, SHARE,
        DIVIDEND, TOTALCAP, postipo=100, filler=0.123456789,
        data_types=[1])

    n_markets = 99#CLOSE.shape[1]
    closes = market_data[:, :n_markets] # exclude cash.
    cm = causal_matrix_ratios(closes, verbose=False, n_neighbors=30,
                              method='nearest', max_data=1000)
    # Add cash
    cm_withcash = np.zeros((cm.shape[0]+1, cm.shape[1]+1))
    cm_withcash[:cm.shape[0], :cm.shape[1]] = cm
    cm_withcash[-1, -1] = 1.

    joblib.dump([settings, cm_withcash], 'saved_data/causality_matrix.pkl')
    return None

def mySettings():
    settings={}
    # Futures Contracts
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20000104'
    settings['endInSample'] = '20131231'

    # Only keep markets that have not died out by beginInSample.
    settings['markets'] = joblib.load('linear_hf/1000_stock_names.pkl')
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__, fname='linear_hf/1000_nyse_stocks.pkl')
    print results['stats']
