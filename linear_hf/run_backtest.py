""" Quantiacs-code based backtester. """
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from linear_hf.preprocessing import preprocess_mini
from linear_hf import training

def _make_empty_datadict(markets, max_time=10000):
    """ Make a dictionary containing placeholders for OPEN, CLOSE,
    HIGH, LOW and DATE data that will be filled up as training continues.
    """
    data = {}
    for key in ['OPEN', 'CLOSE', 'HIGH', 'LOW']:
        data[key] = np.nan * np.ones((max_time, len(markets)))
    data['DATE'] = np.nan * np.ones((max_time))
    return data


def _append_new_data(past_data, OPEN, CLOSE, HIGH, LOW, DATE):
    """ Append the OPEN, ..., DATE data to their respective arrays
    in the past_data dictionary, assuming this is done on each timestep.

    Args:
        past_data (dict): Dictionary containing 'OPEN', ..., 'DATE' as
            keys and nan-padded value arrays as values.
        OPEN, CLOSE, HIGH, LOW, DATE: Quantiacs data arrays.

    Returns:
        OPEN, ..., DATE: Data with pre-pended past entries from the dict.
            The dictionary is also updated with the new data.
    """
    keys = ['OPEN', 'CLOSE', 'HIGH', 'LOW', 'DATE']
    first_nan = np.where(np.isnan(past_data['OPEN']))[0][0]
    lookback = OPEN.shape[0]
    for key in keys:
        if first_nan == 0:
            # This is the first iteration!
            past_data[key][:lookback] = locals()[key]
        else:
            past_data[key][
                first_nan - lookback + 1 : first_nan + 1] = locals()[key]
    if first_nan == 0:
        return [past_data[key][:lookback] for key in keys]
    else:
        return [past_data[key][: first_nan + 1] for key in keys]


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, CLOSE_LASTTRADE,
                    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND,
                    TOTALCAP, exposure, equity, settings, fundEquity):
    OPEN, CLOSE, HIGH, LOW, DATE = _append_new_data(
        settings['past_data'], OPEN, CLOSE, HIGH, LOW, DATE)
    print(OPEN.shape)
    # Preprocess the data
    market_data, all_data, _ = preprocess_mini(
        settings['markets'], OPEN, CLOSE, HIGH, LOW, DATE,
        postipo=100, filler=0.123456789,
        data_types=settings['data_types'])

    # Print progress out.
    print('Iter {} [{}], fundEquity {}, lookback {}.'.format(
        settings['iter'], DATE[-1],
        fundEquity[-1].mean(), all_data.shape[0]))

    # Initialize neural net.
    if settings['iter'] == 0:
        settings = training.init_nn(settings, all_data.shape[1])

    # Train the net.
    if settings['iter'] % settings['retrain_interval'] == 0:
        settings['scaler'] = StandardScaler().fit(all_data)
        all_data = settings['scaler'].transform(all_data)
        if settings['restart_variables']:
            settings['nn'].restart_variables()
        settings = training.train(
            settings=settings, all_data=all_data, market_data=market_data)
    else:
        all_data = settings['scaler'].transform(all_data)

    # Predict a portfolio.
    settings['nn'].load()
    if settings['nn_type'] == 'rnn':
        # Only predict portfolio based on today's prices, as the rnn
        # will carry-through its internal state.
        positions = settings['nn'].predict(all_data[-1:])
    else:
        positions = settings['nn'].predict(all_data[-settings['horizon']:])
    settings['current_positions'] = positions

    # Set positions to zero, cash to 1 if don't trade is True.
    settings['iter'] += 1
    return positions, settings

def mySettings():
    """ Settings for the backtester"""
    settings = {}
    # Futures Contracts
    settings['n_time'] = 30 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 25 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = 1 # L1 regularizer strength.
    settings['num_epochs'] = 2 # Number of epochs each day.
    settings['batch_size'] = 64
    settings['val_period'] = 0
    settings['lr'] = 1e-5 # Learning rate.
    settings['iter'] = 0
    settings['lookback'] = 100
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20010104'
    settings['endInSample'] = '20131231'
    settings['retrain_interval'] = 100
    settings['allow_shorting'] = True
    settings['lr_mult_base'] = 1.
    settings['restart_variables'] = True
    settings['nn_type'] = 'rnn'  # 'linear' or 'rnn'
    settings['nn'] = None
    settings['data_types'] = [1]
    settings['markets'] = joblib.load(
        'linear_hf/tickerData/1000_stock_names.pkl')
    # settings['all_data'] holds an accumulating array 
    # with more and more data.
    settings['past_data'] = _make_empty_datadict(
        settings['markets'])
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    RESULTS = quantiacsToolbox.runts(
        __file__, fname='linear_hf/tickerData/1000_nyse_stocks.pkl')

    print RESULTS['stats']
    RESULTS['nn'] = None
    joblib.dump(RESULTS, 'saved_data/results_of_this_run.pkl')

