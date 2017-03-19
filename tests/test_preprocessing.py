import pytest
import quantiacsToolbox
from quantiacsToolbox import loadData
import sys
import os
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
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
    assert np.sum(filled == np.pi).sum() == 48, 'wrong postipo behavior.'


def test_batching():
    # Make fake data that runs from 1 to 10 for each 'stock'.
    horizon = 7
    n_for_sharpe = 5
    n_valid = 4

    # Set n_data so that number of training points is exactly 128.
    n_data = 128 + n_valid + horizon - 1 + 2 * n_for_sharpe
    all_data = np.ones((n_data, 7)) * np.arange(n_data).reshape(n_data, 1) + 1
    market_data = np.ones((n_data, 4)) * np.arange(n_data).reshape(n_data, 1) + 1

    # Split the dataset into four training batches of size 32 each and
    # a validation batch of size 4.
    all_batches = []
    market_batches = []
    for b_id in range(4):
        all_val, market_val, all_batch, market_batch = preprocessing.split_val_tr(
            all_data, market_data, valid_period=n_valid, batch_size=32,
            batch_id=b_id, horizon=horizon, n_for_sharpe=n_for_sharpe, randseed=1)
        all_batches.append(all_batch)
        market_batches.append(market_batch)

    all_batches = np.vstack(all_batches)
    market_batches = np.vstack(market_batches)

    # Check that the training set covers all data.
    assert_array_equal(np.sort(all_batches[:, 0, 0]),
                       np.arange(128) + 1)
    assert_array_equal(np.sort(market_batches[:, 0, 0]),
                       np.arange(128) + 1 + horizon)

    # Check that the validation set starts where it is supposed to start.
    valid_start = n_data - n_valid - horizon - n_for_sharpe + 1
    assert_array_equal(np.sort(np.unique(all_val[:, 0, 0])),
                       np.arange(4) + 1 + valid_start)
    assert_array_equal(np.sort(np.unique(market_val[:, 0, 0])),
                       np.arange(4) + 1 + horizon + valid_start)
