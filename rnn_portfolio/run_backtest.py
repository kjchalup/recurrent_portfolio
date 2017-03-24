""" Quantiacs-code based backtester. """
import joblib
from rnn_portfolio.preprocessing import preprocess
from rnn_portfolio import training


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE,
                    exposure, equity, settings, fundEquity):
    OPEN, CLOSE, HIGH, LOW, DATE = training.append_new_data(
        settings['past_data'], OPEN, CLOSE, HIGH, LOW, DATE)

    # Preprocess the data
    market_data, all_data, _ = preprocess(
        settings, OPEN, CLOSE, HIGH, LOW, DATE,
        postipo=100, filler=0.123456789)

    # Print pro(re?)gress out.
    print('Iter {} [{}], fundEquity {}, lookback {}.'.format(
        settings['iter'], DATE[-1],
        fundEquity[-1].mean(), all_data.shape[0]))

    # Initialize neural net.
    if settings['iter'] == 0:
        settings = training.init_nn(settings, all_data.shape[1])

    # Train the net.
    if settings['iter'] % settings['retrain_interval'] == 0:
        if settings['restart_variables']:
            settings['nn'].restart_variables()
        settings = training.train(
            settings=settings, all_data=all_data, market_data=market_data)

    # Load the best-validation score nn and predict a portfolio.
    settings['nn'].load()
    positions = settings['nn'].predict(all_data[-1:])
    settings['iter'] += 1
    return positions, settings


def mySettings():
    """ Settings for the backtester"""
    settings = {}
    # Futures Contracts
    settings['n_time'] = 252  # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 252  # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['num_epochs'] = 100  # Number of epochs each day.
    settings['batch_size'] = 32
    settings['val_period'] = 16
    settings['lr'] = 1e-2  # Learning rate.
    settings['iter'] = 0
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20010104'
    settings['endInSample'] = '20131231'
    settings['retrain_interval'] = 1000
    settings['allow_shorting'] = True
    settings['lr_mult_base'] = 1.
    settings['restart_variables'] = True
    settings['nn'] = None
    settings['data_types'] = [1, 4]
    settings['markets'] = ['AAPL', 'GOOG', 'MMM', 'CASH']
    settings['past_data'] = training.make_empty_datadict(settings['markets'])
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    RESULTS = quantiacsToolbox.runts(__file__)

    print RESULTS['stats']
    RESULTS['nn'] = None
    joblib.dump(RESULTS, 'saved_data/results_of_this_run.pkl')

