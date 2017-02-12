import pytest
import quantiacsToolbox
from quantiacsToolbox import loadData
import sys
import os
import pandas as pd
import numpy as np

from context import linear_hf
from linear_hf.preprocessing import non_nan_markets
from linear_hf.preprocessing import preprocess
from linear_hf.preprocessing import nan_markets

def fillnans(inArr):
    ''' fills in (column-wise)value gaps with the most recent non-nan value.

    fills in value gaps with the most recent non-nan value.
    Leading nan's remain in place. The gaps are filled in only after the first non-nan entry.

    Args:
      inArr (list, numpy array)

    Returns:
      returns an array of the same size as inArr with the nan-values replaced by the most recent non-nan entry.
    '''

    inArr=inArr.astype(float)
    nanPos= np.where(np.isnan(inArr))
    nanRow=nanPos[0]
    nanCol=nanPos[1]
    myArr=inArr.copy()
    for i in range(len(nanRow)):
        if nanRow[i] >0:
            myArr[nanRow[i],nanCol[i]]=myArr[nanRow[i]-1,nanCol[i]]
            
    return myArr

beginInSample='20100101'
endInSample='20141231'

@pytest.fixture
def names_with_no_nans():
    names_with_no_nans = non_nan_markets(start_date = beginInSample, end_date = endInSample, postipo = 100, lookback = 0)
    
    #take only the first 50 because otherwise it takes too long
    names_with_no_nans=names_with_no_nans[0:20]
    return names_with_no_nans

@pytest.fixture
def names_with_nans():
    names_with_nans = nan_markets(start_date=beginInSample,
                                end_date=endInSample,
                                postipo=100,
                                lookback=1000)

    names_with_nans = names_with_nans[800:900]
    return names_with_nans


@pytest.fixture
def dataDict(names_with_no_nans):
    # Return dataDict, change dataToLoad to include more fields of data if necessary
    dataDict = loadData(marketList=names_with_no_nans, 
                        beginInSample=beginInSample, 
                        endInSample=endInSample, 
                        dataDir='tickerData', 
                        refresh=False, 
                        dataToLoad=set(['DATE','OPEN','CLOSE','HIGH','LOW'
                                    ,'VOL','RINFO', 'CLOSE_LASTTRADE',
                                    'CLOSE_ASK', 'CLOSE_BID', 'RETURN',
                                    'SHARE', 'DIVIDEND', 'TOTALCAP']))
    return dataDict


@pytest.fixture
def dataDict_withCASH(names_with_nans):
    names_with_nans = names_with_nans + ['CASH']
    # Return dataDict, change dataToLoad to include more fields of data if necessary
    dataDict_withCASH = loadData(marketList=names_with_nans, 
                        beginInSample=beginInSample, 
                        endInSample=endInSample, 
                        dataDir='tickerData', 
                        refresh=False, 
                        dataToLoad=set(['DATE','OPEN','CLOSE','HIGH','LOW'
                                    ,'VOL','RINFO', 'CLOSE_LASTTRADE',
                                    'CLOSE_ASK', 'CLOSE_BID', 'RETURN',
                                    'SHARE', 'DIVIDEND', 'TOTALCAP']))
    dataDict_withCASH['markets']=names_with_nans
    return dataDict_withCASH

@pytest.fixture
def loaded_data(dataDict): 
    # Return open, close, high, low data as np.array
    loaded_data = np.hstack([dataDict['OPEN'],dataDict['CLOSE'],dataDict['HIGH'],dataDict['LOW']])
    return loaded_data

def test_no_nans(loaded_data):
    assert((np.isnan(loaded_data)).sum()==0)

def test_no_zeros(loaded_data):
    assert((loaded_data==0).sum()==0)

def test_no_crazy_returns(dataDict):
    nMarkets = dataDict['CLOSE'].shape[1]
    sessionReturnTemp = np.append( np.empty((1,nMarkets))*np.nan,(( dataDict['CLOSE'][1:,:]- dataDict['OPEN'][1:,:]) / dataDict['CLOSE'][0:-1,:] ), axis =0 ).copy()
    sessionReturn=np.nan_to_num( fillnans(sessionReturnTemp) )
    gapsTemp=np.append(np.empty((1,nMarkets))*np.nan, (dataDict['OPEN'][1:,:]- dataDict['CLOSE'][:-1,:].astype(float)) / dataDict['CLOSE'][:-1:],axis=0)
    gaps=np.nan_to_num(fillnans(gapsTemp))

    # check if a default slippage is specified
    slippage_setting = 0.05
    slippageTemp = np.append(np.empty((1,nMarkets))*np.nan, ((dataDict['HIGH'][1:,:] - dataDict['LOW'][1:,:]) / dataDict['CLOSE'][:-1,:] ), axis=0) * slippage_setting
    SLIPPAGE = np.nan_to_num(fillnans(slippageTemp))
    

    # Check that daily returns are less than 0.5 for session, gap, return. gap+session-slippage should be approximate returns for any given stock
    assert (abs(SLIPPAGE)>0.7).sum()==0
    assert (abs(gaps)>2).sum()==0
    assert (abs(sessionReturn)>2).sum()==0
    assert (abs(gaps)+abs(sessionReturn)>2).sum()==0
    assert (abs(gaps)==np.inf).sum()==0
    assert (abs(sessionReturn)==np.inf).sum()==0
    assert (abs(SLIPPAGE)==np.inf).sum()==0


def test_preprocessing_nonans_nozeros(dataDict_withCASH):
    dataDict = dataDict_withCASH
    # Pass the data to preprocessing
    filler = 0.123456789
    postipo = 100
    markets = dataDict['markets']
    opens = dataDict['OPEN']
    closes = dataDict['CLOSE']
    highs = dataDict['HIGH']
    lows = dataDict['LOW']
    vols = dataDict['VOL']
    dates = dataDict['DATE']
    close_lasttrade = dataDict['CLOSE_LASTTRADE']
    close_ask = dataDict['CLOSE_ASK']
    close_bid = dataDict['CLOSE_BID']
    returns = dataDict['RETURN']
    shares = dataDict['SHARE']
    dividends = dataDict['DIVIDEND']
    totalcaps = dataDict['TOTALCAP']
    
    n_markets4 = len(dataDict['markets'])*4
    n_markets = n_markets4/4
    nMarkets = n_markets
    sessionReturnTemp = np.append( np.empty((1,nMarkets))*np.nan,(( dataDict['CLOSE'][1:,:]- dataDict['OPEN'][1:,:]) / dataDict['CLOSE'][0:-1,:] ), axis =0 ).copy()
    sessionReturn=np.nan_to_num( fillnans(sessionReturnTemp) )
    gapsTemp=np.append(np.empty((1,nMarkets))*np.nan, (dataDict['OPEN'][1:,:]- dataDict['CLOSE'][:-1,:].astype(float)) / dataDict['CLOSE'][:-1:],axis=0)
    gaps=np.nan_to_num(fillnans(gapsTemp))

    # check if a default slippage is specified
    slippage_setting = 0.05
    slippageTemp = np.append(np.empty((1,nMarkets))*np.nan, ((dataDict['HIGH'][1:,:] - dataDict['LOW'][1:,:]) / dataDict['CLOSE'][:-1,:] ), axis=0) * slippage_setting
    SLIPPAGE = np.nan_to_num(fillnans(slippageTemp))

    # Check before
    assert (abs(SLIPPAGE)>0.7).sum()==0
    #assert (abs(gaps)>6).sum()==0
    assert (abs(sessionReturn)>6).sum()==0
    #assert (abs(gaps)+abs(sessionReturn)>10).sum()==0
    assert (abs(gaps)==np.inf).sum()==0
    assert (abs(sessionReturn)==np.inf).sum()==0
    assert (abs(SLIPPAGE)==np.inf).sum()==0


    filled_prices, all_data, should_retrain = preprocess(markets, opens, closes, 
                                                        highs, lows, vols, dates, 
                                                        close_lasttrade, close_ask, 
                                                        close_bid, returns, 
                                                        shares, dividends, totalcaps, 
                                                        postipo=postipo, filler=filler,
                                                        data_types=[0,1,2,3,4,5,6,7,8,9,10,11,12])
    
    n_markets4 = filled_prices.shape[1]
    assert (filled_prices==0).sum()==0, "Zeros put into preprocessor!"
    assert (np.isnan(filled_prices)).sum()==0, "NaNs put into preprocess!"
    assert np.mod((filled_prices==filler).sum(),filled_prices.shape[0]*4)==0, "Filler is only applied to dead stocks!"   
    
    dataDict['OPEN'] = filled_prices[:,:n_markets]
    dataDict['CLOSE'] = filled_prices[:,n_markets:2*n_markets]
    dataDict['HIGH'] = filled_prices[:,n_markets*2:n_markets*3]
    dataDict['LOW'] = filled_prices[:,n_markets*3:n_markets*4]
    
    sessionReturnTemp = np.append( np.empty((1,nMarkets))*np.nan,(( dataDict['CLOSE'][1:,:]- dataDict['OPEN'][1:,:]) / dataDict['CLOSE'][0:-1,:] ), axis =0 ).copy()
    sessionReturn=np.nan_to_num( fillnans(sessionReturnTemp) )
    gapsTemp=np.append(np.empty((1,nMarkets))*np.nan, (dataDict['OPEN'][1:,:]- dataDict['CLOSE'][:-1,:].astype(float)) / dataDict['CLOSE'][:-1:],axis=0)
    gaps=np.nan_to_num(fillnans(gapsTemp))

    # check if a default slippage is specified
    slippage_setting = 0.05
    slippageTemp = np.append(np.empty((1,nMarkets))*np.nan, ((dataDict['HIGH'][1:,:] - dataDict['LOW'][1:,:]) / dataDict['CLOSE'][:-1,:] ), axis=0) * slippage_setting
    SLIPPAGE = np.nan_to_num(fillnans(slippageTemp))
    

    # Check after
    assert (abs(SLIPPAGE)>0.7).sum()==0
    #assert (abs(gaps)>6).sum()==0
    assert (abs(sessionReturn)>6).sum()==0
    #assert (abs(gaps)+abs(sessionReturn)>6).sum()==0
    assert (abs(gaps)==np.inf).sum()==0
    assert (abs(sessionReturn)==np.inf).sum()==0
    assert (abs(SLIPPAGE)==np.inf).sum()==0


