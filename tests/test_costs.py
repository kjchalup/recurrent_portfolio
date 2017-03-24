import pytest
from quantiacsToolbox import loadData
import numpy as np

from context import linear_hf
from linear_hf.quantiacs_code import quantiacs_calculation
from linear_hf.costs import compute_np_sharpe
from linear_hf.costs import compute_tf_sharpe

@pytest.fixture
def create_data():
    """ Load stock data from a ticker file. """
    markets = ['fake1']
    data_dict = loadData(
        marketList=markets, beginInSample='20090101', endInSample='20130101',
        dataToLoad=set(['DATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']),
        dataDir='tickerData', refresh=False)

    market_data = np.hstack([data_dict['OPEN'], data_dict['CLOSE'],
                             data_dict['HIGH'], data_dict['LOW']])

    all_data = np.array(market_data)

    settings = {'markets': markets,
                'lookback': 2,
                'slippage': 0.05}

    return data_dict, all_data, market_data, settings


def compare_returns(data_dict, positions, settings, market_data):
    """ Check that quantiacs and our calculation of returns agree. """
    # Compute returns using Quantiacs code.
    rs_qc = quantiacs_calculation(data_dict, positions,
                                  settings)['returns'].sum(axis=1)

    # Compute returns using our Numpy code.
    pos = positions[None, :-1, :]
    prices = market_data[None, 1:, :]
    rs_np = compute_np_sharpe(
        positions=pos, prices=prices, return_rs=True)[0, :]

    # Check if the returns agree, more-less.
    rs_ratio = rs_np[6:] / rs_qc[7:]
    rs_ratio = rs_ratio[np.isfinite(rs_ratio)]
    assert np.all(rs_ratio <= 1.05) and np.all(rs_ratio > .95)


def test_np_quantiacs_agree_all_ones(create_data):
    data_dict, _, market_data, settings = create_data
    n_timesteps, n_markets = market_data.shape
    n_markets = n_markets / 4
    positions_equal = np.ones([n_timesteps, n_markets]) / float(n_markets)
    compare_returns(data_dict=data_dict, positions=positions_equal,
                    settings=settings, market_data=market_data)


def test_np_quantiacs_agree_randpos(create_data):
    data_dict, _, market_data, settings = create_data
    n_timesteps, n_markets = market_data.shape
    n_markets = n_markets / 4
    np.random.seed(0)
    positions_rand = np.random.rand(n_timesteps, n_markets) - 0.5
    compare_returns(data_dict=data_dict, positions=positions_rand,
                    settings=settings, market_data=market_data)


def test_tf_sharpe(create_data):
    data_dict, _, market_data, settings = create_data

    # Prepare the positions.
    n_timesteps, n_markets = market_data.shape
    n_markets = n_markets / 4
    np.random.seed(0)
    positions_rand = np.random.rand(n_timesteps, n_markets) - 0.5

    # Prepare the data.
    timelen = 50
    pos_short = positions_rand[-timelen:, :]
    poss = positions_rand[None, -timelen:-1, :]
    prices = market_data[None, -timelen+1:, :]
    
    for data_key in ['OPEN', 'CLOSE', 'HIGH', 'LOW', 'DATE']:
        data_dict[data_key] = data_dict[data_key][-timelen:]
    sharpe_qc = quantiacs_calculation(
        data_dict, pos_short, settings)['stats']['sharpe']
    sharpe_tf = compute_tf_sharpe(positions=poss, prices=prices)
    sharpe_np = compute_np_sharpe(positions=poss, prices=prices)

    tf_qc_ratio = sharpe_tf / float(sharpe_qc)
    qc_np_ratio = sharpe_qc / float(sharpe_np)
    np_tf_ratio = sharpe_np / float(sharpe_tf)

    assert tf_qc_ratio > 0.95 and tf_qc_ratio < 1.05,\
        "TFSharpe / Quantiacs Sharpe don't agree to 5%"
    assert qc_np_ratio > 0.95 and qc_np_ratio < 1.05,\
        "Quantiacs Sharpe / NP Sharpe don't agree to 5%"
    assert np_tf_ratio > 0.95 and np_tf_ratio < 1.05,\
        "Numpy Sharpe and TF Sharpe don't agree to 5%"
