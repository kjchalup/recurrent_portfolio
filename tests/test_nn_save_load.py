import numpy as np
from numpy.testing import assert_array_almost_equal, assert_raises

from context import linear_hf
from linear_hf import neuralnet
from linear_hf import NP_DTYPE


def test_nn_restart_variables():
    n_ftrs = 4
    n_markets = 2
    n_time = 10
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1
    batch_in = np.ones((1, n_time, n_ftrs), dtype=NP_DTYPE)
    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)

    out1 = nn.predict(batch_in[0, -horizon:])
    nn.restart_variables()
    out2 = nn.predict(batch_in[0, -horizon:])
    assert np.sum(out1 != out2) > 0

    
def test_nn_save_and_load():
    n_ftrs = 4
    n_markets = 2
    n_time = 10
    n_sharpe = 3
    horizon = n_time - n_sharpe + 1
    batch_in = np.ones((1, n_time, n_ftrs), dtype=NP_DTYPE)

    nn = neuralnet.Linear(n_ftrs, n_markets, n_time, n_sharpe)
    out1 = nn.predict(batch_in[0, -horizon:])
    nn.save()
    nn.restart_variables()
    nn.load()
    out3 = nn.predict(batch_in[0, -horizon:])
    assert_array_almost_equal(out1, out3)
