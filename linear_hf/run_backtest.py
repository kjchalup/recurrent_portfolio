""" Quantiacs-code based backtester. """
import joblib
from linear_hf.preprocessing import preprocess
from linear_hf import training

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, CLOSE_LASTTRADE,
                    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND,
                    TOTALCAP, exposure, equity, settings, fundEquity):
    n_markets = len(settings['markets'])

    # Preprocess the data
    market_data, all_data, _ = preprocess(
        settings['markets'][:n_markets], OPEN[:, :n_markets],
        CLOSE[:, :n_markets], HIGH[:, :n_markets],
        LOW[:, :n_markets], VOL[:, :n_markets], DATE,
        CLOSE_LASTTRADE[:, :n_markets], CLOSE_ASK[:, :n_markets], CLOSE_BID[:, :n_markets],
        RETURN[:, :n_markets], SHARE[:, :n_markets],
        DIVIDEND[:, :n_markets], TOTALCAP[:, :n_markets], postipo=100, filler=0.123456789,
        data_types=settings['data_types'])

    # Print progress out.
    print('Iter {} [{}], fundEquity {}.'.format(
        settings['iter'], DATE[-1], fundEquity[-1].mean()))

    # Initialize neural net.
    if settings['iter'] == 0:
        settings = training.init_nn(settings, all_data.shape[1])

    # Train the net.
    if settings['iter'] % settings['retrain_interval'] == 0:
        if settings['restart_variables']:
            settings['nn'].restart_variables()
        settings = training.train(
            settings=settings, all_data=all_data, market_data=market_data)

    # Predict a portfolio.
    settings['nn'].load()
    if settings['nn_type'] == 'rnn':
        positions = settings['nn'].predict(all_data[-settings['n_time']:])
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
    settings['n_sharpe'] = 10 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = 1 # L1 regularizer strength.
    settings['num_epochs'] = 30 # Number of epochs each day.
    settings['batch_size'] = 32
    settings['val_period'] = 0
    settings['lr'] = 1e-5 # Learning rate.
    settings['dont_trade'] = False # If on, don't trade.
    settings['iter'] = 0
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20000104'
    settings['endInSample'] = '20131231'
    settings['retrain_interval'] = 100
    settings['allow_shorting'] = True
    settings['lr_mult_base'] = 1.
    settings['restart_variables'] = True
    settings['nn_type'] = 'linear' # 'linear' or 'rnn'
    settings['nn'] = None
    settings['data_types'] = [1]
    settings['markets'] = joblib.load('linear_hf/tickerData/1000_stock_names.pkl')
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    RESULTS = quantiacsToolbox.runts(__file__, fname='linear_hf/tickerData/1000_nyse_stocks.pkl')

    print RESULTS['stats']
    joblib.dump(RESULTS, 'saved_data/results_of_this_run.pkl')

