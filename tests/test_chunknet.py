import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from context import linear_hf
from linear_hf import chunknet

@pytest.fixture
def make_data():
    n_batch = 17
    n_markets = 20
    n_time = 33
    n_sharpe = 7
    batch_in = np.ones((n_batch, n_time, n_markets * 4),
                       dtype=np.float32)
    return batch_in, n_sharpe

def test_w_init():
    n_ftrs = 100
    n_time = 7
    n_sharpe = 3
    n_markets = 100
    n_blocks = 5

    weights = chunknet.initialize_blockdiagonal(
        n_ftrs, n_time, n_sharpe, n_markets, n_blocks)
    assert len(weights) == n_blocks
    for weight in weights:
        shape = tuple([float(int(tf_dim))
                       for tf_dim in weight.get_shape()])
        assert shape == (n_ftrs * (n_time - n_sharpe + 1) /
                         float(n_blocks), n_markets / float(n_blocks))

def test_w_init_errors():
    n_ftrs = 100
    n_time = 7
    n_sharpe = 3
    n_markets = 100
    n_blocks = 5
    with pytest.raises(ValueError):
        ws = chunknet.initialize_blockdiagonal(n_ftrs, n_time, n_sharpe,
                                               n_markets, 7)
    with pytest.raises(ValueError):
        ws = chunknet.initialize_blockdiagonal(7777,
                                               n_time, n_sharpe,
                                               n_markets, n_blocks)
    with pytest.raises(ValueError):
        ws = chunknet.initialize_blockdiagonal(n_ftrs, n_time, n_sharpe,
                                               n_markets, 10**5)

def test_nn_all_inputs_ones(make_data):
    batch_in, n_sharpe = make_data
    _, n_time, n_ftrs = batch_in.shape
    n_markets = n_ftrs / 4
    horizon = n_time - n_sharpe + 1
    n_chunks = 5
    W_init = chunknet.initialize_blockdiagonal(n_ftrs, n_time, n_sharpe,
                                               n_markets, n_chunks)
    W_init = [W * 0 + 1. for W in W_init]
    nn = chunknet.ChunkLinear(n_ftrs, n_markets, n_time,
                              n_sharpe, n_chunks, W_init)
    assert_array_almost_equal(nn.predict(batch_in[0, -horizon:]),
                              np.ones(n_markets, dtype=np.float32) /
                              float(n_markets))
