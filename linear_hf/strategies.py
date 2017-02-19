""" Portfolio strategies to be used as base strategies for the neural nets."""
import tensorflow as tf

from linear_hf import NP_DTYPE

def long_equal(in_prices):
    """ Long all stocks, equally.

    Args:
      in_prices (n_batch, horizon, n_markets): Equity prices used
        to predict the portfolios.

    Returns:
      positions (n_sharpe, n_markets): All ones/n_markets.
    """
    positions = in_prices * 0. + 1.
    positions /= tf.reduce_sum(tf.abs(positions), axis=2, keep_dims=True)
    return positions[:, -1, :]

def cumulative_returns(in_prices):
    """ Long or short stocks proportionally to their returns over the
    `horizon` period.

    Args:
      in_prices (n_batch, horizon, n_markets): Equity prices used
        to predict the portfolios.

    Returns:
      positions (n_sharpe, n_markets): All ones/n_markets.
    """
    rets = (in_prices[:, -1, :] - in_prices[:, -2, :]) / in_prices[:, -2, :]
    # logrets_neg = -tf.log(tf.abs(tf.minimum(rets, 0))
    # logrets_pos = tf.log(tf.maximum(rets, 0) + 1e-7)
    # positions = logrets_pos + logrets_neg
    positions = rets
    return positions / tf.reduce_sum(tf.abs(positions), axis=1,
                                     keep_dims=True)
