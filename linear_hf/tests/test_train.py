import pytest
import os
import sys
import datetime
import numpy as np
import pandas as pd

import inspect 
#Include scripts from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from preprocessing import non_nan_markets
from quantiacsToolbox import loadData
import neuralnet


beginInSample = '20060101'
endInSample = '20151231'

@pytest.fixture
def names_with_no_nans():
    names_with_no_nans = non_nan_markets(start_date = beginInSample, end_date = endInSample, postipo = 100, lookback=0)
    names_with_no_nans=names_with_no_nans[0:10]
    return names_with_no_nans

@pytest.fixture
def dataDict(names_with_no_nans):
    # Return dataDict, change dataToLoad to include more fields of data if necessary
    dataDict = loadData(marketList = names_with_no_nans, beginInSample = '20060101', endInSample = '20151231', dataDir = 'tickerData', refresh = False, dataToLoad = set(['DATE','OPEN','CLOSE','HIGH','LOW','VOL','RINFO']))
    return dataDict

@pytest.fixture
def loaded_data(dataDict):
    # Return open, close, high, low data as np.array
    loaded_data = np.hstack([dataDict['OPEN'],dataDict['CLOSE'],dataDict['HIGH'],dataDict['LOW']])
    return loaded_data

def test_gradient_step(loaded_data):
    stocks = loaded_data
    total_time_steps, n_markets = stocks.shape
    n_time = 200
    n_sharpe = 32
    n_batch = 128
    nn = neuralnet.Linear(n_markets=n_markets, 
                          n_time=n_time, 
                          n_sharpe=n_sharpe)
    batch_in = stocks[:-1]
    batch_out = stocks[-n_sharpe:]
    nn_loss_before = nn.loss_np(batch_in=batch_in, batch_out=batch_out)
    nn.train_step(lr=1e-10, batch_in=batch_in, batch_out=batch_out)
    nn_loss_after = nn.loss_np()

    assert nn_loss_after <  nn_loss_before, 'Loss did not decrease.'

    
