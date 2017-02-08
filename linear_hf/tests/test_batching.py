import pytest
import quantiacsToolbox
from quantiacsToolbox import loadData
#from quantiacsToolbox import fillnans
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
import batching_splitting
from batching_splitting import split_validation_training

valid_period = 2
horizon = 165
n_for_sharpe = 32
batch_id = 0
batch_size = 128
randseed = 0
beginInSample='20090101'
endInSample='20141231'

names_with_no_nans = non_nan_markets(start_date = beginInSample, end_date = endInSample, postipo = 100, lookback = 0)
    
names_with_no_nans=names_with_no_nans[0:7]

dataDict = loadData(marketList=names_with_no_nans, 
                beginInSample=beginInSample,    
                endInSample=endInSample, 
                dataDir='tickerData', 
                refresh=False, 
                dataToLoad=set(['DATE','OPEN','CLOSE','HIGH','LOW','VOL','RINFO']))
market_data = np.hstack([dataDict['OPEN'],dataDict['CLOSE'],dataDict['HIGH'],dataDict['LOW']])
all_data = np.hstack([dataDict['OPEN'],dataDict['CLOSE'],dataDict['HIGH'],dataDict['LOW']])

def turnoff_test_training_validation_separation(all_data=all_data, 
                                        market_data=market_data,
                                        valid_period=valid_period, 
                                        horizon=horizon, 
                                        n_for_sharpe=n_for_sharpe, 
                                        batch_id=batch_id, 
                                        batch_size=batch_size, 
                                        randseed=randseed):
    count = 0
    all_val, market_val, all_batch, market_batch = split_validation_training(
                                                all_data=all_data,
                                                market_data=market_data, 
                                                valid_period=valid_period, 
                                                horizon=horizon,
                                                n_for_sharpe=n_for_sharpe, 
                                                batch_id=batch_id, 
                                                batch_size=batch_size, 
                                                randseed= randseed)
    
    if market_val is not None:
        for val_0 in range(market_val.shape[0]):
            for val_t in range(market_val.shape[1]):
                for batch_0 in range(market_batch.shape[0]):
                    for batch_t in range(market_batch.shape[1]):
                        agree=(market_val[val_0,val_t,:]==market_batch[batch_0,batch_t,:]).sum()
                        if agree==market_val.shape[2]:
                            count += 1
    assert count == 0,'If this number is larger than 0, training data is bleeding into validation data'

def test_market_data_indexing(all_data=all_data, 
                    market_data=market_data, 
                    valid_period=valid_period, 
                    horizon=horizon,
                    n_for_sharpe=n_for_sharpe,
                    batch_id=batch_id, 
                    batch_size=batch_size, 
                    randseed=randseed):
    all_val, market_val, all_batch, market_batch = split_validation_training(all_data=all_data,
                                        market_data=market_data, 
                                        valid_period=valid_period, 
                                        horizon=horizon,
                                        n_for_sharpe=n_for_sharpe, 
                                        batch_id=batch_id, 
                                        batch_size=batch_size, 
                                        randseed=randseed)
    n_markets4 = market_batch.shape[2]
    for i in range(n_for_sharpe-1):
        assert (market_batch[0,-i-2,:n_markets4]==all_batch[0,-i-1,:n_markets4]).sum()==n_markets4, "market_batch and all_batch are not correctly indexed against each other in batches!"
    if valid_period>0:
        for i in range(n_for_sharpe-1):
            assert (market_val[0,-i-2,:n_markets4]==all_val[0,-i-1,:n_markets4]).sum()==n_markets4, "market_val and all_val are not correctly indexed against each other in validation batches!"

def test_zero_nan_in_any_batch(all_data=all_data, 
                    market_data=market_data, 
                    valid_period=valid_period, 
                    horizon=horizon,
                    n_for_sharpe=n_for_sharpe,
                    batch_id=batch_id, 
                    batch_size=batch_size, 
                    randseed=randseed):
    all_val, market_val, all_batch, market_batch = split_validation_training(all_data=all_data,
                                        market_data=market_data, 
                                        valid_period=valid_period, 
                                        horizon=horizon,
                                        n_for_sharpe=n_for_sharpe, 
                                        batch_id=batch_id, 
                                        batch_size=batch_size, 
                                        randseed=randseed)
    
    if all_val is not None:
        assert (all_val==0).sum()==0
        assert (market_val==0).sum()==0
        assert (np.isnan(all_val)).sum()==0
        assert (np.isnan(market_val)).sum()==0
    
    
    assert((all_batch==0).sum()==0)
    assert((market_batch==0).sum()==0)
    assert((np.isnan(all_batch)).sum()==0)
    assert((np.isnan(market_batch)).sum()==0)
    
    
def test_batch_coverage(all_data=all_data, 
                    market_data=market_data, 
                    valid_period=valid_period, 
                    horizon=horizon,
                    n_for_sharpe=n_for_sharpe,
                    batch_id=batch_id, 
                    batch_size=batch_size, 
                    randseed=randseed):
    all_val, market_val, all_batch, market_batch = split_validation_training(all_data=all_data,
                                        market_data=market_data, 
                                        valid_period=valid_period, 
                                        horizon=horizon,
                                        n_for_sharpe=n_for_sharpe, 
                                        batch_id=batch_id, 
                                        batch_size=batch_size, 
                                        randseed=randseed)
    firstpt_batch = all_data[0,:]
    lastpt_batch = all_data[-valid_period-n_for_sharpe-2,:]
    firstpt_valid = all_data[-valid_period-horizon-n_for_sharpe+1,:]
    lastpt_valid = all_data[-1,:]
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    for z in range(int(np.floor((all_data.shape[0]-horizon-2*n_for_sharpe-valid_period+1)/float(batch_size)))):
        all_val, market_val, all_batch, market_batch = split_validation_training(all_data=all_data,
                                        market_data=market_data, 
                                        valid_period=valid_period, 
                                        horizon=horizon,
                                        n_for_sharpe=n_for_sharpe, 
                                        batch_id=z, 
                                        batch_size=batch_size, 
                                        randseed=randseed)
        for j in range(all_batch.shape[0]):
            for k in range(all_batch.shape[1]):
                div_sum=np.divide(all_batch[j,k,:],firstpt_batch).sum()
                if (div_sum>all_batch.shape[2]-0.00005 and div_sum<all_batch.shape[0]+0.00005):
                    flag1 = True
                div_sum=np.divide(all_batch[j,k,:],lastpt_batch).sum()
                if (div_sum>all_batch.shape[2]-0.00005 and div_sum<all_batch.shape[0]+0.00005):
                    flag2 = True
        if valid_period > 0:
            for j in range(all_val.shape[0]):
                for k in range(all_val.shape[1]):
                    div_sum=np.divide(all_val[j,k,:],firstpt_valid).sum()
                    if (div_sum>all_val.shape[2]-0.00005 and div_sum<all_val.shape[0]+0.00005):
                        flag3 = True
                    div_sum=np.divide(all_val[j,k,:],lastpt_valid).sum()
                    if (div_sum>all_val.shape[2]-0.00005 and div_sum<all_val.shape[0]+0.00005):
                        flag4 = True

        if all_val is not None:
            assert (all_val==0).sum()==0
            assert (market_val==0).sum()==0
            assert (np.isnan(all_val)).sum()==0
            assert (np.isnan(market_val)).sum()==0
    
    
        assert((all_batch==0).sum()==0)
        assert((market_batch==0).sum()==0)
        assert((np.isnan(all_batch)).sum()==0)
        assert((np.isnan(market_batch)).sum()==0)
    
    assert flag1, "No coverage of first point in batch data!"
    assert flag2, "No coverage of last point in batch data!"
    assert flag3, "No coverage of first point in validation data!"
    assert flag4, "No coverage of last point in validation data!"
