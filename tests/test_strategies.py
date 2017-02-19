from context import linear_hf
from linear_hf import strategies
from linear_hf import TF_DTYPE
from linear_hf import NP_DTYPE

import numpy as np
from numpy.testing import assert_array_almost_equal
import tensorflow as tf

def test_long_equal():
    n_batch = 3
    horizon = 7
    n_markets = 5
    in_prices_tf = tf.placeholder(dtype=TF_DTYPE,
                                  shape=[n_batch, horizon, n_markets])
    in_prices = np.ones((n_batch, horizon, n_markets), dtype=NP_DTYPE)
    sess = tf.Session()
    positions = sess.run(strategies.long_equal(in_prices_tf),
                         {in_prices_tf: in_prices})
    sess.close()
    long_equal = np.ones((n_batch, n_markets)) / n_markets
    assert_array_almost_equal(positions, long_equal)

def test_cumulative_returns():
    n_batch = 1
    horizon = 7
    n_markets = 5
    in_prices_tf = tf.placeholder(dtype=TF_DTYPE,
                                  shape=[n_batch, horizon, n_markets])
    in_prices = np.ones((n_batch, horizon, n_markets), dtype=NP_DTYPE)
    in_prices[:, -1, 0] = .5
    in_prices[:, -1, -1] = 2
    sess = tf.Session()
    positions = sess.run(strategies.cumulative_returns(in_prices_tf),
                         {in_prices_tf: in_prices})
    sess.close()
    long_equal = np.ones((n_batch, n_markets)) + 1e-7
    long_equal[:, 0] += -.5
    long_equal[:, -1] += 1
    # long_equal = np.log(long_equal)

    long_equal /= np.sum(np.abs(long_equal), axis=1, keepdims=True)
    assert_array_almost_equal(positions, long_equal)
