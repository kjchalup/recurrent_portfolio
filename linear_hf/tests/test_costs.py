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

beginInSample='20090101'
endInSample='20141231'
names_with_no_nans = non_nan_markets(start_date=beginInSample, end_date=endInSample, postipo=100, lookback=0)
names_with_no_nans = names_with_no_nans[0:100]
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
    pos = positions[None,:,:]
    prices = market_data[None,:,:]

    # Calculate daily numpy sharpe returns
    rs_numpy = compute_numpy_sharpe(positions=pos, prices=prices, slippage=0.05, return_returns = True)
    
    # Remove singleton dimension to directly compare two values.
    rs_np = rs_numpy[0,:]

    # Calculate daily returns ratio between numpy and backtester, should not deviate more than 3%!
    daily_returns_ratio = np.divide(rs_np[2:],rs_qc[2:])
    for num in daily_returns_ratio:
        assert( num <= 1.03 and num >= 0.97, "Deviation between numpy returns and backtester returns is less than 3% every day" )
    
    # Calculate sharpe ratio for numpy, quantiacs, and neural net!
    sharpe_np = compute_numpy_sharpe(positions=pos, prices=prices, slippage=0.05)
    sharpe_qc = return_qc['stats']['sharpe']
    ratio_sharpe = sharpe_np/float(sharpe_qc)
    assert( ratio_sharpe > 0.95 and ratio_sharpe < 1.05, "Sharpe ratio of numpy and qc off by more than 5%")

def test_costfn_backtester_all1s(dataDict = dataDict, positions=positions_all1, settings=settings, market_data = market_data):
    evaluate_systems(dataDict = dataDict, positions=positions_all1, settings=settings, market_data=market_data)

def test_costfn_backtester_randpos(dataDict = dataDict, positions=positions_rand, settings=settings, market_data=market_data):
    evaluate_systems(dataDict=dataDict, positions=positions_rand, settings=settings, market_data=market_data)


'''
def test_nn_random_init():
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

returns = quantiacs_calculation(dataDict, positions, settings)
rs_qc = returns['returns'].sum(axis=1)
pos = positions[None,:,:]
prices = market_data[None,:,:]
rs_np = compute_numpy_sharpe(positions=pos, prices=prices, slippage=0.05, return_returns = True)
import pdb;pdb.set_trace()
'''
