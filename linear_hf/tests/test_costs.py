import pytest
import quantiacsToolbox
from quantiacsToolbox import loadData
import sys
import os
import pandas as pd
import numpy as np
import inspect

#Include scripts from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from preprocessing import non_nan_markets
from quantiacs_code import quantiacs_calculation
from costs import compute_numpy_sharpe
from costs import compute_sharpe_tf

beginInSample='20090101'
endInSample='20141231'
names_with_no_nans = non_nan_markets(start_date=beginInSample, end_date=endInSample, postipo=100, lookback=0)
names_with_no_nans = names_with_no_nans[200:250]
dataDict = loadData(marketList=names_with_no_nans, beginInSample=beginInSample, endInSample=endInSample, dataDir='tickerData', refresh=False, dataToLoad=set(['DATE', 'OPEN', 'CLOSE', 'HIGH', 'LOW']))
market_data = np.hstack([dataDict['OPEN'], dataDict['CLOSE'], dataDict['HIGH'], dataDict['LOW']])
all_data = np.hstack([dataDict['OPEN'], dataDict['CLOSE'], dataDict['HIGH'], dataDict['LOW']])
 
settings={'markets':names_with_no_nans,
              'lookback': 2,
              'slippage': 0.05}


n_timesteps, n_markets = market_data.shape
n_markets = n_markets/4
positions_all1 = np.ones([n_timesteps, n_markets])/float(n_markets)
positions_rand = np.random.rand(n_timesteps, n_markets)-0.5

    

def evaluate_systems(dataDict, positions, settings, market_data):
    # Calculate QC returns, sum the returns across stocks to get daily returns. 
    
    return_qc = quantiacs_calculation(dataDict, positions, settings)
    rs_qc = return_qc['returns'].sum(axis=1)
    
    # Add singletone dimension to prices and positions to mimic having batches.
    pos = positions[None,:-1,:]
    prices = market_data[None,1:,:]

    # Calculate daily numpy sharpe returns
    rs_numpy = compute_numpy_sharpe(positions=pos, prices=prices, slippage=0.05, return_returns = True)
    
    # Remove singleton dimension to directly compare two values.
    rs_np = rs_numpy[0,:]

    # Calculate daily returns ratio between numpy and backtester, should not deviate more than 3%!
    
    

    daily_returns_ratio = np.divide(rs_np[6:],rs_qc[7:])
    for num in daily_returns_ratio:
        if num>1.05 or num<0.95:
            import pdb;pdb.set_trace()
        assert num <= 1.05 and num>=0.95
    # Calculate sharpe ratio for numpy, quantiacs, and neural net!
    sharpe_np = compute_numpy_sharpe(positions=pos, prices=prices, slippage=0.05)
    sharpe_qc = return_qc['stats']['sharpe']
    
    ratio_sharpe = sharpe_np/float(sharpe_qc)
    assert ratio_sharpe > 0.85 and ratio_sharpe < 1.15, "Sharpe ratio of numpy and qc off by more than 5%"

def test_costfn_backtester_all1s(dataDict = dataDict, positions=positions_all1, settings=settings, market_data = market_data):
    evaluate_systems(dataDict = dataDict, positions=positions_all1, settings=settings, market_data=market_data)

def test_costfn_backtester_randpos(dataDict = dataDict, positions=positions_rand, settings=settings, market_data=market_data):
    evaluate_systems(dataDict=dataDict, positions=positions_rand, settings=settings, market_data=market_data)

def test_tf_sharpe_using_premade_positions(position=positions_rand, batch_out=market_data, dataDict=dataDict, settings=settings):
    num_days_to_calc = 50
    pos_short = positions_rand[-num_days_to_calc:,:]
    poss = positions_rand[None,-num_days_to_calc:-1,:]

    market_short = market_data[-num_days_to_calc:,:]
    pricess = market_data[None,-num_days_to_calc+1:,:]
    
    dataDict['OPEN']=dataDict['OPEN'][-num_days_to_calc:]
    dataDict['CLOSE']=dataDict['CLOSE'][-num_days_to_calc:]
    dataDict['HIGH']=dataDict['HIGH'][-num_days_to_calc:]
    dataDict['LOW']=dataDict['LOW'][-num_days_to_calc:]
    dataDict['DATE']=dataDict['DATE'][-num_days_to_calc:]
    dataDict['RINFO']=dataDict['RINFO'][-num_days_to_calc:]
    return_qc = quantiacs_calculation(dataDict, pos_short, settings)
    sharpe_tf = compute_sharpe_tf(batch_in=poss, batch_out=pricess)
    sharpe_np = compute_numpy_sharpe(positions=poss, prices=pricess, slippage=0.05)
    sharpe_qc = return_qc['stats']['sharpe']
    
    tf_qc_ratio = sharpe_tf/float(sharpe_qc)
    qc_np_ratio = sharpe_qc/float(sharpe_np)
    np_tf_ratio = sharpe_np/float(sharpe_tf)

    assert tf_qc_ratio > 0.95 and tf_qc_ratio < 1.05, "TF Sharpe and Quantiacs Sharpe don't agree to 5%"
    assert qc_np_ratio > 0.95 and qc_np_ratio < 1.05, "Quantiacs Sharpe and Numpy Sharpe don't agree to 5%"
    assert np_tf_ratio > 0.95 and np_tf_ratio < 1.05, "Numpy Sharpe and TF Sharpe don't agree to 5%"

'''
    ret_np = compute_numpy_sharpe(positions=pos, prices=prices, slippage=0.05, return_returns = True)
    rs_np = ret_np[0,:]
    rs_qc = return_qc['returns'].sum(axis=1)
'''


def temp_test_nn_random_init():
    np.random.rand(1)
    n_timesteps, n_ftrs = market_data.shape
    n_markets = n_ftrs/4
    n_sharpe = n_timesteps-2
    batch_in = market_data[None,:-1,:]
    batch_out = market_data[None,1:,:]
    n_batch = 1
    n_time = n_timesteps-1
    
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe, lbd=0)
    nn_loss_before = nn.los_np(batch_in=batch_in, batch_out = batch_out)

