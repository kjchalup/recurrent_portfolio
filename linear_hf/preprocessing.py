import os
import glob
import random
import joblib
import numpy as np

from  datetime import datetime, timedelta


def load_nyse_markets(start_date, end_date, postipo=100, lookback=0):
    """ Loads nyse markets which start before start_date-postipo, and end after start_date.

    Args:
        start_date: date of starting to consider data
        end_date: not used
        postipo: number of days stock befor start_date the stock must start to be considered.
        lookback: not used

    """

    # Load nyse stocks that IPDd between start and end date, with margin padding.
    all_nyse = glob.glob('tickerData/*nyse.txt')
    alives = []
    # Get end_date minus some a number!
    start_date = (datetime.strptime(start_date, '%Y%m%d') +
        timedelta(days=lookback)).strftime('%Y%m%d')

    end_date_minuspostipo = (datetime.strptime(end_date, '%Y%m%d') -
        timedelta(days=postipo)).strftime('%Y%m%d')
    start_date_minuspostipo = (datetime.strptime(start_date, '%Y%m%d') -
        timedelta(days=postipo)).strftime('%Y%m%d')
    for fname in all_nyse:
        f = open(fname, 'r').readlines()
        # Only include stocks that IPO at least 100 days before we begin trading,
        # and that are still alive on that day.
        if (int(f[1].split(',')[0]) < int(start_date_minuspostipo) and 
            int(f[-1].split(',')[0]) > int(start_date)):
            alives.append(fname)
    return [symbol.split('/')[1][:-4] for symbol in alives] 


def non_nan_markets(start_date, end_date, postipo=100, lookback=0):
    """ Stock names with no nans

    Args:
        start_date: start date for which stocks must begin by, adjusted by postipo
        end_date: stocks must live until this date
        postipo: number of days before start_date a stock must start by.
        lookback: not used.

    Returns:
        Names of stocks which fit the above criteria
    """
    
    # Load all stocks that IPDd between start and end date, with margin padding.
    all_nyse = glob.glob('tickerData/*.txt')
    alives = []
    # Get end_date minus some a number!
    start_date = (datetime.strptime(start_date, '%Y%m%d') +
        timedelta(days=lookback)).strftime('%Y%m%d')

    end_date_minuspostipo = (datetime.strptime(end_date, '%Y%m%d') -
        timedelta(days=postipo)).strftime('%Y%m%d')
    start_date_minuspostipo = (datetime.strptime(start_date, '%Y%m%d') -
        timedelta(days=postipo)).strftime('%Y%m%d')
    for fname in all_nyse:
        f = open(fname, 'r').readlines()
        # Only include stocks that IPO at least 100 days before we begin trading,
        # and that are still alive on that day.
        #import pdb;pdb.set_trace()
        if (int(f[1].split(',')[0]) < int(start_date_minuspostipo) and 
            int(f[-1].split(',')[0]) >= int(end_date)):
            
            if len([s for s in f if 'NaN' in s]) == 0:
                alives.append(fname)
    print ('Found '+str(len(alives))+' stocks with no nans starting after '+start_date_minuspostipo)
    return [symbol.split('/')[1][:-4] for symbol in alives] 

