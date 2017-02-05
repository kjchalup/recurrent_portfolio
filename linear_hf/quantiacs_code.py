import numpy as np
import pandas as pd
from test_preprocessing import fillnans
from quantiacsToolbox import stats

def quantiacs_calculation(dataDict, positions, settings):
    """ Evaluates trading returns using quantiacsToolbox code

    Args:
        dataDict: dict from quantiacs' loadData function, needs open, close, high, low
        positions: positions vector (n_timesteps, n_markets), doesn't need to be normalized
        settings: dict with lookback, slippage set. minimum for lookback = 2

    Returns:
        returns: dict with 'fundEquity'(n_timesteps, 1), 'returns' (n_timesteps, n_markets)
    """


    print 'Evaluating Trading System'

    nMarkets=len(settings['markets'])
    endLoop=len(dataDict['DATE'])

    if 'RINFO' in dataDict:
        Rix= dataDict['RINFO'] != 0
    else:
        dataDict['RINFO'] = np.zeros(np.shape(dataDict['CLOSE']))
        Rix = np.zeros(np.shape(dataDict['CLOSE']))

    #%dataDict['exposure']=np.zeros((endLoop,nMarkets))
    dataDict['exposure']=positions
    position = positions
    dataDict['equity']=np.ones((endLoop,nMarkets))
    dataDict['fundEquity'] = np.ones((endLoop,1))
    realizedP = np.zeros((endLoop, nMarkets))
    returns = np.zeros((endLoop, nMarkets))

    sessionReturnTemp = np.append( np.empty((1,nMarkets))*np.nan,(( dataDict['CLOSE'][1:,:]- dataDict['OPEN'][1:,:]) / dataDict['CLOSE'][0:-1,:] ), axis =0 ).copy()
    sessionReturn=np.nan_to_num( fillnans(sessionReturnTemp) )
    gapsTemp=np.append(np.empty((1,nMarkets))*np.nan, (dataDict['OPEN'][1:,:]- dataDict['CLOSE'][:-1,:]-dataDict['RINFO'][1:,:].astype(float)) / dataDict['CLOSE'][:-1:],axis=0)
    gaps=np.nan_to_num(fillnans(gapsTemp))

    slippageTemp = np.append(np.empty((1,nMarkets))*np.nan, ((dataDict['HIGH'][1:,:] - dataDict['LOW'][1:,:]) / dataDict['CLOSE'][:-1,:] ), axis=0) * settings['slippage']
    SLIPPAGE = np.nan_to_num(fillnans(slippageTemp))
    
    startLoop = settings['lookback'] - 1
    # Loop through trading days
    for t in range(startLoop,endLoop):
        todaysP= dataDict['exposure'][t-1,:]
        yesterdaysP = realizedP[t-2,:]
        deltaP=todaysP-yesterdaysP

        newGap=yesterdaysP * gaps[t,:]
        newGap[np.isnan(newGap)]= 0

        newRet = todaysP * sessionReturn[t,:] - abs(deltaP * SLIPPAGE[t,:])
        newRet[np.isnan(newRet)] = 0

        returns[t,:] = newRet + newGap
        dataDict['equity'][t,:] = dataDict['equity'][t-1,:] * (1+returns[t,:])
        dataDict['fundEquity'][t] = (dataDict['fundEquity'][t-1] * (1+np.sum(returns[t,:])))

        realizedP[t-1,:] = dataDict['CLOSE'][t,:] / dataDict['CLOSE'][t-1,:] * dataDict['fundEquity'][t-1] / dataDict['fundEquity'][t] * todaysP

        position[np.isnan(position)] = 0
        position = np.real(position)
        position = position/np.sum(abs(position))
        position[np.isnan(position)] = 0  # extra nan check in case the positions sum to zero

        #dataDict['exposure'][t,:] = position.copy()

    marketRets = np.float64(dataDict['CLOSE'][1:,:] - dataDict['CLOSE'][:-1,:] - dataDict['RINFO'][1:,:])/dataDict['CLOSE'][:-1,:]
    marketRets = fillnans(marketRets)
    marketRets[np.isnan(marketRets)] = 0
    marketRets = marketRets.tolist()
    a = np.zeros((1,nMarkets))
    a = a.tolist()
    marketRets = a + marketRets

    ret={}
    ret['returns'] = np.nan_to_num(returns)

    #ret['tsName']=tsName
    ret['fundDate']=dataDict['DATE']
    ret['fundEquity']=dataDict['fundEquity']
    ret['marketEquity']= dataDict['equity']
    ret['marketExposure'] = dataDict['exposure']
    ret['settings']=settings
    ret['evalDate']=dataDict['DATE'][t]
    ret['stats']=stats(dataDict['fundEquity'])
    return ret
