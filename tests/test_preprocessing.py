import pytest
import quantiacsToolbox
from quantiacsToolbox import loadData
import sys
import os
import pandas as pd
import numpy as np

from context import linear_hf
from linear_hf import preprocessing

BEGIN_DATE = '20100101'
END_DATE = '20141231'

def test_preproc_postipo():
    markets = ['fake']
    dates = np.atleast_2d([20000101, 20000102, 20000103, 20000104,
                           20000105, 20000106, 20000107, 20000108,
                           20000109, 20000110, 20000111, 20000112]).T
    prices = np.ones((12, 1))
    prices[:3] = np.nan
    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=1)
    assert np.isnan(filled).sum() == 0

    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=10,
                                            filler=np.pi)
    assert np.sum(filled == np.pi) == 48, 'Edge case postipo did not work.'

    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=11,
                                            filler=np.pi)
    assert np.sum(filled == np.pi) == 48, 'Normal case postipo did not work.'

    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=100,
                                            filler=np.pi)
    assert np.sum(filled == np.pi) == 48, 'Edge case postipo did not work.'


def test_preproc_nanfill():
    markets = ['fake']
    dates = np.atleast_2d([20000101, 20000102, 20000103, 20000104,
                           20000105, 20000106, 20000107, 20000108,
                           20000109, 20000110, 20000111, 20000112]).T
    prices = np.ones((12, 1))
    prices[5:] = np.nan
    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=1)
    assert np.isnan(filled).sum() == 0, 'nans were not filled.'

def test_preproc_postipo_and_nan():
    markets = ['fake']
    dates = np.atleast_2d([20000101, 20000102, 20000103, 20000104,
                           20000105, 20000106, 20000107, 20000108,
                           20000109, 20000110, 20000111, 20000112]).T
    prices = np.ones((12, 1))
    prices[:3] = np.nan
    prices[5:] = np.nan

    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=1)
    assert np.isnan(filled).sum() == 0, 'nans were not filled.'

    filled, _, _ = preprocessing.preprocess(markets, prices, prices, prices, prices,
                                            prices, dates, prices, prices, prices,
                                            prices, prices, prices, prices, postipo=100,
                                            filler=np.pi)
    assert np.sum(filled==np.pi).sum() == 48, 'wrong postipo behavior.'
