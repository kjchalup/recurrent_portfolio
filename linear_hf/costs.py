import os
import sys
import joblib

import numpy as np
import tensorflow as tf

def compute_numpy_sharpe(positions, prices, slippage=0.05, return_returns = False, n_ignore=2):
    """ Compute average Sharpe ratio of a strategy using Numpy.

    Args:
      positions (n_batch, n_sharpe, n_markets): Portfolio over
        n_sharpe timesteps over n_batch batches.
      price (n_batch, n_sharpe, n_markets * 4): Stock prices
        corresponding to portfolio positions over the same time.
        Should contain (in order) open, close, high and low prices.
      slippage (float): slippage coefficient.
      n_ignore (int): ignore this many of the first returns 
        (to avoid boundary effects breaking training).
    
    Returns:
      sharpe (float): Sharpe ratio that positions achieve, averaged
        over all the batches.
    """
    n_batch, n_sharpe, n_markets = positions.shape
    rs = np.zeros((n_batch, n_sharpe))
    os = prices[:, :, :n_markets]
    cs = prices[:, :, n_markets:2*n_markets]
    hs = prices[:, :, 2*n_markets:3*n_markets]
    ls = prices[:, :, 3*n_markets:4*n_markets]

    for i in range(2, n_sharpe): 
        elem1 = (((os[:, i, :] - cs[:, i-1, :]) * positions[:, i-1, :])/
                 (cs[:, i-2, :] * (1 + rs[:, i-1:i])))
        elem2 = ((cs[:, i, :] - os[:, i, :]) *
                 positions[:, i, :]/cs[:, i-1, :])
        elem3 = hs[:, i, :] - ls[:, i, :]
        elem4 = (positions[:, i, :]/cs[:, i-1, :] - 
                  positions[:, i-1, :]/
                 (cs[:, i-2, :] * (1 + rs[:, i-1:i])))
        rs[:, i] = (elem1 + elem2 -
                    slippage*np.abs(elem3 * elem4)).sum(axis=1)
    rs = rs[:, n_ignore:]
    n_sharpe -= n_ignore
    if return_returns:
        return rs

    return ((np.prod(rs+1, axis=1)**(252./n_sharpe)-1) /
            (np.sqrt(252 * ((rs**2).sum(axis=1) / n_sharpe -
            np.sum(rs, axis=1)**2 / n_sharpe**2)))).mean()

def sharpe_tf(positions, prices, n_sharpe, n_markets, slippage=.05, n_ignore=2, cost='sharpe'):
    """ Compute average Sharpe ratio of a strategy using Tensorflow.

    Args:
      positions (n_batch, n_sharpe, n_markets): Tensor, portfolio over
        n_sharpe timesteps over n_batch batches.
      price (n_batch, n_sharpe, n_markets * 4): Tensor, stock prices
        corresponding to portfolio positions over the same time.
        Should contain (in order) open, close, high and low prices.
      n_sharpe (float): number of timesteps that Sharpe is calculated
        over (must match tf.shape(positions)[1]).
      n_markets (float): number of markets. Must match
        tf.shape(positions)[2].
      slippage (float): slippage coefficient.
      n_ignore (int): ignore this many of the first returns
        (to avoid boundary effects breaking training).
      cost (str): cost to use: 'sharpe', 'min_return', 'mean_return',
        'mixed_return' or 'sortino'.

    Returns:
      sharpe (tf.float): Tensor, representation of the Sharpe
        ratio that positions achieve, averaged over all the batches.
    """

    n_batch = tf.shape(positions)[0]
    rs_list = [tf.zeros((n_batch, 1)), tf.zeros((n_batch, 1))]
    os = prices[:, :, :n_markets]
    cs = prices[:, :, n_markets:2*n_markets]
    hs = prices[:, :, 2*n_markets:3*n_markets]
    ls = prices[:, :, 3*n_markets:4*n_markets]
   
    for i in range(2, n_sharpe):
        elem1 = (((os[:, i, :] - cs[:, i-1, :]) * positions[:, i-1, :])/
                 (cs[:, i-2, :] * (1 + rs_list[i-1])))
        elem2 = ((cs[:, i, :] - os[:, i, :]) *
                 positions[:, i, :]/cs[:, i-1, :])
        elem3 = hs[:, i, :] - ls[:, i, :]
        elem4 = (positions[:, i, :]/cs[:, i-1, :] -
                 positions[:, i-1, :]/
                 (cs[:, i-2, :] * (1 + rs_list[i-1])))
        rs_list.append(tf.reduce_sum(
            elem1 + elem2 - slippage*np.abs(elem3 * elem4),
            axis=1, keep_dims=True))
    rs = tf.stack(rs_list, axis=1)[:, n_ignore:, 0]
    n_sharpe -= n_ignore

    prod_rs = tf.reduce_prod(rs + 1, axis=1)
    if cost == 'sharpe':
        return tf.reduce_mean((tf.pow(prod_rs, (252./n_sharpe))-1) /
                (tf.sqrt(252 * (tf.reduce_sum(tf.pow(rs, 2), axis=1) / n_sharpe -
                tf.pow(tf.reduce_sum(rs, axis=1), 2) / n_sharpe**2))))
    elif cost == 'sortino':
        pos_rets = tf.minimum(rs, 0)
        pos_std = (tf.sqrt(252 * (tf.reduce_sum(
            tf.pow(pos_rets, 2), axis=1) / n_sharpe -
            tf.pow(tf.reduce_sum(pos_rets, axis=1), 2) / n_sharpe**2)))
        return tf.reduce_mean((tf.pow(prod_rs, (252./n_sharpe))-1) / (pos_std + 1e-7))
    elif cost == 'min_return':
        return tf.reduce_min(prod_rs)
    elif cost == 'mean_return':
        return tf.reduce_mean(prod_rs)
    elif cost == 'mixed_return':
        return tf.reduce_min(prod_rs) + tf.reduce_mean(prod_rs)

def compute_sharpe_tf(batch_in, batch_out):
    n, n_time, n_ftrs = batch_in.shape
    n, n_sharpe, n_markets4 = batch_out.shape
    n_markets = n_markets4/4
    
    sess = tf.Session()
    batch_in_tf = tf.placeholder(
        tf.float32, shape=[None, n_time, n_ftrs], 
        name='input_batch')
    batch_out_tf = tf.placeholder(
        tf.float32, shape=[None, n_sharpe, n_markets * 4],
        name='output_batch')

    return sess.run(sharpe_tf(batch_in_tf, batch_out_tf, n_sharpe, n_markets), 
                    {batch_in_tf: batch_in, batch_out_tf: batch_out})
