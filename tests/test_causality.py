import os
import sys
import numpy as np

from context import linear_hf
from linear_hf.causality import compute_independence
from linear_hf.causality import iscause_anm

def test_independent_uniform_variables_are_independent():
    x = np.random.rand(1000, 1)
    y = np.random.rand(1000, 1)
    pval = compute_independence(x, y)
    print(pval)
    assert compute_independence(x, y) > 1e-2

def test_independent_normal_and_uniform_are_independent():
     x = np.random.rand(10000, 1)
     y = np.random.randn(10000, 1)
     assert compute_independence(x, y) > 1e-2

def test_complex_independent_vars_are_independent():
    x = np.sin(np.random.randn(1000, 1)) * 3 + 1
    y = np.exp(np.random.rand(1000, 1))**2 - 10
    assert compute_independence(x, y) > 1e-2

def test_dependent_vars_are_dependent_additive_noise():
    x = np.random.rand(1000, 1)
    y = x + np.random.rand(1000, 1) * 1e-4
    assert compute_independence(x, y) < 1e-2

def test_dependent_vars_are_dependent_multiplicative_noise():
    x = np.random.randn(10000, 1)
    y = x * np.random.rand(10000, 1) * 1e-4
    assert compute_independence(x, y) < 1e-2

def test_additive_noise_model():
    x = np.random.rand(1000, 1)
    y = np.sin(10*x) + np.random.rand(1000, 1)
    xy_pval, yx_pval = iscause_anm(x, y)
    assert xy_pval > 1e-2
    assert yx_pval < 1e-2
