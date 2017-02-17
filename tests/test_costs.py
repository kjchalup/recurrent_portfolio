import pytest
import quantiacsToolbox
from quantiacsToolbox import loadData
import pandas as pd
import numpy as np

#Include scripts from parent directory
from context import linear_hf
from linear_hf.preprocessing import non_nan_markets
from linear_hf.quantiacs_code import quantiacs_calculation
from linear_hf.costs import compute_numpy_sharpe
from linear_hf.costs import compute_sharpe_tf
from linear_hf import neuralnet

beginInSample = '20090101'
endInSample = '20141231'
names_with_no_nans = non_nan_markets(start_date=beginInSample,
                                     end_date=endInSample, postipo=100, lookback=0)
names_with_no_nans = names_with_no_nans[200:250]
dataDict = loadData(marketList=names_with_no_nans,
                    beginInSample=beginInSample,
                    endInSample=endInSample,
                    dataDir='tickerData', refresh=False,
                    dataToLoad=set(['DATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']))
market_data = np.hstack([dataDict['OPEN'], dataDict['CLOSE'],
                         dataDict['HIGH'], dataDict['LOW']])
all_data = np.hstack([dataDict['OPEN'], dataDict['CLOSE'],
                      dataDict['HIGH'], dataDict['LOW']])
 
settings={'markets': names_with_no_nans,
          'lookback': 2,
          'slippage': 0.05}

n_timesteps, n_markets = market_data.shape
n_markets = n_markets / 4
positions_all1 = np.ones([n_timesteps, n_markets]) / float(n_markets)
np.random.seed(0)
positions_rand = np.random.rand(n_timesteps, n_markets) - 0.5

def evaluate_systems(dataDict, positions, settings, market_data):
    # Calculate QC returns, sum the returns across stocks to get daily returns. 

    return_qc = quantiacs_calculation(dataDict, positions, settings)
    rs_qc = return_qc['returns'].sum(axis=1)

    # Add singletone dimension to prices and positions to mimic having batches.
    pos = positions[None, :-1, :]
    prices = market_data[None, 1:, :]

    # Calculate daily numpy sharpe returns
    rs_numpy = compute_numpy_sharpe(positions=pos,
                                    prices=prices,
                                    slippage=0.05,
                                    return_returns=True,
                                    n_ignore=0)
    # Remove singleton dimension to directly compare two values.
    rs_np = rs_numpy[0, :]
    # Calculate daily returns ratio between numpy and backtester, 
    # should not deviate more than 3%!
    daily_returns_ratio = np.divide(rs_np[6:], rs_qc[7:])
    for num in daily_returns_ratio:
        assert num <= 1.05 and num >= 0.95
    # Calculate sharpe ratio for numpy, quantiacs, and neural net!
    sharpe_np = compute_numpy_sharpe(positions=pos,
                                     prices=prices,
                                     slippage=0.05,
                                     n_ignore=0)
    sharpe_qc = return_qc['stats']['sharpe']
    
    ratio_sharpe = sharpe_np / float(sharpe_qc)
    # Sharpe from Quantiacs and sharpe from numpy differ by more than 15%
    assert ratio_sharpe > 0.85 and ratio_sharpe < 1.15

def test_costfn_backtester_all1s(dataDict=dataDict,
                                 positions=positions_all1,
                                 settings=settings,
                                 market_data=market_data):
    evaluate_systems(dataDict=dataDict,
                     positions=positions_all1,
                     settings=settings,
                     market_data=market_data)

def test_costfn_backtester_randpos(dataDict=dataDict,
                                   positions=positions_rand,
                                   settings=settings,
                                   market_data=market_data):
    evaluate_systems(dataDict=dataDict,
                     positions=positions_rand,
                     settings=settings,
                     market_data=market_data)

def test_tf_sharpe_using_premade_positions(position=positions_rand,
                                           batch_out=market_data,
                                           dataDict=dataDict,
                                           settings=settings):
    num_days_to_calc = 50
    pos_short = positions_rand[-num_days_to_calc:, :]
    poss = positions_rand[None, -num_days_to_calc:-1, :]

    market_short = market_data[-num_days_to_calc:, :]
    pricess = market_data[None, -num_days_to_calc+1:, :]
    
    dataDict['OPEN'] = dataDict['OPEN'][-num_days_to_calc:]
    dataDict['CLOSE'] = dataDict['CLOSE'][-num_days_to_calc:]
    dataDict['HIGH'] = dataDict['HIGH'][-num_days_to_calc:]
    dataDict['LOW'] = dataDict['LOW'][-num_days_to_calc:]
    dataDict['DATE'] = dataDict['DATE'][-num_days_to_calc:]
    dataDict['RINFO'] = dataDict['RINFO'][-num_days_to_calc:]
    return_qc = quantiacs_calculation(dataDict, pos_short, settings)
    sharpe_tf = compute_sharpe_tf(batch_in=poss, batch_out=pricess)
    sharpe_np = compute_numpy_sharpe(positions=poss, prices=pricess, slippage=0.05)
    sharpe_qc = return_qc['stats']['sharpe']

    tf_qc_ratio = sharpe_tf/float(sharpe_qc)
    qc_np_ratio = sharpe_qc/float(sharpe_np)
    np_tf_ratio = sharpe_np/float(sharpe_tf)

    assert tf_qc_ratio > 0.95 and tf_qc_ratio < 1.05, "TFSharpe / Quantiacs Sharpe don't agree to 5%"
    assert qc_np_ratio > 0.95 and qc_np_ratio < 1.05, "Quantiacs Sharpe / NP Sharpe don't agree to 5%"
    assert np_tf_ratio > 0.95 and np_tf_ratio < 1.05, "Numpy Sharpe and TF Sharpe don't agree to 5%"


def test_random_init_nn_sharpe():
    beginInSample = '20141201'
    endInSample = '20141231'
    names_with_no_nans = non_nan_markets(start_date=beginInSample,
                                         end_date=endInSample, postipo=100, lookback=0)
    names_with_no_nans = names_with_no_nans[:5]
    dataDict = loadData(marketList=names_with_no_nans,
                        beginInSample=beginInSample,
                        endInSample=endInSample,
                        dataDir='tickerData', refresh=False,
                        dataToLoad=set(['DATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']))
    market_data = np.hstack([dataDict['OPEN'], dataDict['CLOSE'],
                             dataDict['HIGH'], dataDict['LOW']])
    all_data = np.hstack([dataDict['OPEN'], dataDict['CLOSE'],
                          dataDict['HIGH'], dataDict['LOW']])

    settings = {'markets':names_with_no_nans,
                'lookback': 2,
                'slippage': 0.05}

    n_timesteps, n_markets = market_data.shape
    n_markets = n_markets/4
    positions_all1 = np.ones([n_timesteps, n_markets])/float(n_markets)
    np.random.seed(0)
    positions_rand = np.random.rand(n_timesteps, n_markets)-0.5

    np.random.rand(1)
    n_timesteps, n_ftrs = market_data.shape
    n_markets = n_ftrs/4
    n_sharpe = n_timesteps-1
    batch_in = market_data[None, :-1, :]
    batch_out = market_data[None, 1:, :]
    n_batch = 1
    n_time = n_timesteps-1

    # Check if everything matches when lambda = 0
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=0, allow_shorting=False)
    nn_pos = nn._positions_np(batch_in=batch_in)
    nn_loss = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn_l1 = nn.l1_penalty_np()
    sharpe_nn_0 = -(nn_loss - nn_l1)
    sharpe_tf_0 = compute_sharpe_tf(batch_in=nn_pos, batch_out=batch_out)
    sharpe_np_0 = compute_numpy_sharpe(positions=nn_pos, prices=batch_out, slippage=0.05)

    ratio_nn_tf = sharpe_nn_0/float(sharpe_tf_0)
    ratio_tf_np = sharpe_tf_0/float(sharpe_np_0)
    ratio_np_nn = sharpe_np_0/float(sharpe_nn_0)

    assert ratio_nn_tf < 1.03 and ratio_nn_tf > 0.97, "Neural net loss doesn't agree with TF Sharpe"
    assert ratio_tf_np < 1.03 and ratio_tf_np > 0.97, "TF Sharpe doesn't agree with NP Sharpe"
    assert ratio_np_nn < 1.03 and ratio_tf_np > 0.97, "NP Sharpe doesn't agree with NN Loss"

    # Check if everything matches when lambda = 11
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=11, allow_shorting=False)
    nn_pos = nn._positions_np(batch_in=batch_in)
    nn_loss = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn_l1 = nn.l1_penalty_np()

    sharpe_nn = -(nn_loss - nn_l1)
    sharpe_tf = compute_sharpe_tf(batch_in=nn_pos, batch_out=batch_out)
    sharpe_np = compute_numpy_sharpe(positions=nn_pos, prices=batch_out, slippage=0.05)

    ratio_nn_tf = sharpe_nn/float(sharpe_tf)
    ratio_tf_np = sharpe_tf/float(sharpe_np)
    ratio_np_nn = sharpe_np/float(sharpe_nn)

    assert ratio_nn_tf < 1.03 and ratio_nn_tf > 0.97, "Neural net loss doesn't agree with TF Sharpe"
    assert ratio_tf_np < 1.03 and ratio_tf_np > 0.97, "TF Sharpe doesn't agree with NP Sharpe"
    assert ratio_np_nn < 1.03 and ratio_tf_np > 0.97, "NP Sharpe doesn't agree with NN Loss"
    # Re-initialize network with cost argument to get min_return back out from loss.
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=11,
                          allow_shorting=False, cost='min_return')
    nn_pos = nn._positions_np(batch_in=batch_in)
    nn_loss = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn_l1 = nn.l1_penalty_np()
    # Multiplly the output min_return by -1 to account for -1 inside the nn.
    nn_min_return = -1 * (nn_loss - nn_l1)

    np_returns = np.prod(1 + compute_numpy_sharpe(
        positions=nn_pos, prices=batch_out,
        slippage=0.05, return_returns=True), axis=1)
    np_min_return = np_returns.min()
    rat_np_nn = np_min_return / nn_min_return
    assert rat_np_nn < 1.01 and rat_np_nn > 0.99, "Min return cost function is broken!"

    # Re-initialize network with cost argument to get mean_return back out from loss.
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=11,
                          allow_shorting=False, cost='mean_return')
    nn_pos = nn._positions_np(batch_in=batch_in)
    nn_loss = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn_l1 = nn.l1_penalty_np()
    nn_mean_return = -1 * (nn_loss - nn_l1)
    np_returns = np.prod(1 + compute_numpy_sharpe(positions=nn_pos,
        prices=batch_out, slippage=0.05, return_returns=True), axis=1)
    np_mean_return = np_returns.mean()
    rat_np_nn = np_mean_return / nn_mean_return
    assert rat_np_nn < 1.01 and rat_np_nn > 0.99, "Mean return cost function is broken!"

    # Re-initialize network with cost argument to get mixed_return back out from loss.
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=11,
                          allow_shorting=False, cost='mixed_return')
    nn_pos = nn._positions_np(batch_in=batch_in)
    nn_loss = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn_l1 = nn.l1_penalty_np()
    nn_mixed_return = -1 * (nn_loss - nn_l1)
    np_returns = np.prod(1 + compute_numpy_sharpe(
        positions=nn_pos, prices=batch_out, slippage=0.05,
                                      return_returns=True), axis=1)
    np_mixed_return = np_returns.mean() +  np_returns.min()
    rat_np_nn = np_mixed_return / nn_mixed_return
    assert rat_np_nn < 1.05 and rat_np_nn > 0.95, "Mixed return cost function is broken!"

    # Re-initialize network with cost argument to sortino to check loss
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=11,
                          allow_shorting=False, cost='sortino')
    nn_pos = nn._positions_np(batch_in=batch_in)
    nn_loss = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn_l1 = nn.l1_penalty_np()
    nn_sortino_return = -1 * (nn_loss - nn_l1)
    np_returns = compute_numpy_sharpe(positions=nn_pos, 
                                      prices=batch_out, slippage=0.05,
                                      return_returns=True)

    # Standard deviation is: np.std on daily returns * sqrt(252) to annualize
<<<<<<< HEAD
    denominator = (np.std(np_returns[np_returns < 0.0])) * np.sqrt(252) + 1e-7
    numerator = np.prod(np_returns+1)**(252. / (n_sharpe-2))-1
    np_sortino_return = numerator / denominator
    rat_np_nn = np_sortino_return / nn_sortino_return
    assert rat_np_nn < 1.05 and rat_np_nn > 0.95, "Sortino cost function is broken!"




=======
    pos_rets = np.array(np_returns)
    pos_rets[pos_rets > 0] = 0.
    denominator = np.sqrt(252 * (np.sum(pos_rets**2, axis=1) /
        (n_sharpe-2) - np.sum(pos_rets, axis=1)**2 / (n_sharpe-2)**2))

    numerator = np.prod(np_returns+1)**(252. / (n_sharpe-2))-1
    np_sortino_return = numerator / denominator
    rat_np_nn = np_sortino_return / nn_sortino_return
    assert rat_np_nn < 1.15 and rat_np_nn > 0.85, "Sortino cost function is broken!"
>>>>>>> 400dea6bc4eb23f286b757e02c26b191a75b3990
