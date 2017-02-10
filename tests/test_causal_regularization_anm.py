import os
import sys
import time
import numpy as np

import pytest

from context import linear_hf
from linear_hf.causality import causal_matrix
# from linear_hf.causality import estimate_causality

@pytest.fixture
def make_data():
    fs = [lambda x: np.sin(x*10), 
          lambda x: np.cos(10*x),
          lambda x: -np.sin(10*x),
          lambda x: -np.cos(10*x)]
    xs = np.random.rand(1000, 4)
    ys = np.zeros((1000, 4))
    for y_id in range(4):
        ys[:, y_id] = fs[y_id](xs[:, y_id]) + np.random.rand(1000)

    return np.hstack([xs, ys])

def test_causal_matrix(make_data):
    cm, t = causal_matrix(make_data)
    import pdb; pdb.set_trace()
    should_be_causal = np.array([cm[0, 4], cm[1, 5], cm[2, 6], cm[3, 7]]) 
    assert (should_be_causal < 1e-2).sum() == 0, 'Some of the causal relationships were not detected.'
    shouldnt_be_causal = np.array([cm[0,1], cm[1, 2], cm[2, 3], cm[3, 4], 
                                  cm[4, 5], cm[5, 6], cm[6, 7], cm[0, 5],
                                  cm[1, 7], cm[2, 5], cm[5, 1], cm[6, 2]])
    assert (shouldnt_be_causal > 1e-2).sum() == 0, 'Some of the causal relationships were not detected.'

