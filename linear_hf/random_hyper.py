"""Supply hyperparameters for the neural net to test."""
import sys
import random

import numpy as np
import joblib

import linear_hf.neuralnet as neuralnet
from linear_hf.preprocessing import non_nan_markets
from linear_hf.batching_splitting import split_validation_training

def supply(keys):
    """Take a dictionary of keys and ranges of parameters and return a
random selection from those ranges."""
    pass
