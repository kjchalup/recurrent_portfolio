""" Cost functions to be optimized by the neural networks. """
import numpy as np
import tensorflow as tf

from . import TF_DTYPE

def sharpe_tf(positions, prices, slippage=.05, n_ignore=0):
    """ Define (in Tensorflow) the Sharpe ratio of a strategy.

    Args:
        positions (n_batch, n_sharpe, n_markets): Tensor, portfolio over
          n_sharpe timesteps over n_batch batches.
        price (n_batch, n_sharpe, n_markets * 4): Tensor, stock prices
          corresponding to portfolio positions over the same time.
          Should contain (in order) open, close, high and low prices.
        slippage (float): slippage coefficient.
        n_ignore (int): ignore this many of the first returns
          (to avoid boundary effects breaking training).

    Returns:
        sharpe (tf.float): Tensor, representation of the Sharpe
          ratio that positions achieve, averaged over all the batches.
    """
    n_batch, n_sharpe, n_markets = positions.get_shape().as_list()
    n_batch = tf.shape(positions)[0]
    rs_list = [tf.zeros((n_batch, 1), dtype=TF_DTYPE),
               tf.zeros((n_batch, 1), dtype=TF_DTYPE)]
    os, cs, hs, ls = _extract_data(prices, n_markets)

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
    sharpe = tf.reduce_min(
        (tf.pow(prod_rs, (252. / n_sharpe))-1) /
        (tf.sqrt(252 * (tf.reduce_sum(tf.pow(rs, 2), axis=1) / n_sharpe -
                        tf.pow(tf.reduce_sum(rs, axis=1), 2) / n_sharpe**2))))
    return sharpe


def compute_tf_sharpe(positions, prices, slippage=.05, n_ignore=0):
    sess = tf.Session()
    pos_tf = tf.placeholder(tf.float32, shape=positions.shape)
    price_tf = tf.placeholder(tf.float32, shape=prices.shape)
    return sess.run(sharpe_tf(pos_tf, price_tf, slippage, n_ignore),
                    {pos_tf: positions, price_tf: prices})


def compute_np_sharpe(
        positions, prices, slippage=0.05, n_ignore=0, return_rs=False):
    """ Compute average Sharpe ratio of a strategy using Numpy.

    This is mostly useful for debugging Tensorflow and Quantiacs code.

    Args:
        positions (n_batch, n_sharpe, n_markets): Portfolio over
          n_sharpe timesteps over n_batch batches.
        price (n_batch, n_sharpe, n_markets * 4): Stock prices
          corresponding to portfolio positions over the same time.
          Should contain (in order) open, close, high and low prices.
        slippage (float): slippage coefficient.
        return_rs (bool): If yes, return the full
          returns matrix instead of the cost.
        n_ignore (int): ignore this many of the first returns
          (to avoid boundary effects breaking training).

    Returns:
        sharpe (float): Sharpe ratio that positions achieve, averaged
          over all the batches.
    """
    n_batch, n_sharpe, n_markets = positions.shape
    rs = np.zeros((n_batch, n_sharpe))
    os, cs, hs, ls = _extract_data(prices, n_markets)

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
    if return_rs:
        return rs
    else:
        return np.min((np.prod(rs + 1, axis=1)**(252. / n_sharpe)-1) /
                      (np.sqrt(252 * ((rs**2).sum(axis=1) / n_sharpe - np.sum(
                          rs, axis=1)**2 / n_sharpe**2))))


def _extract_data(prices, n_markets):
    """ Extract the open, close, high and low prices from the price matrix. """
    os = prices[:, :, :n_markets]
    cs = prices[:, :, n_markets:2*n_markets]
    hs = prices[:, :, 2*n_markets:3*n_markets]
    ls = prices[:, :, 3*n_markets:4*n_markets]
    return os, cs, hs, ls
