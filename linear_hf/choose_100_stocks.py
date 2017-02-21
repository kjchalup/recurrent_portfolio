"""Choose 100 stocks for use in the causality/noncausal experiments."""
import sys
import joblib
import numpy as np

def choose_100_stocks(seed):
    """Choose 100 of the 1000 stock names."""
    # Choose the seed.
    np.random.seed(seed) # pylint: disable=no-member
    allstocks = joblib.load('linear_hf/1000_stock_names.pkl')
    # Use numpy so that we can use the boolean mask.
    stocks = np.array(allstocks)[
        np.random.choice(1000, 99, replace=False)] # pylint: disable=no-member
    # Make it a normal list of strings plus CASH.
    return [stock for stock in stocks] + ['CASH']

if __name__ == '__main__':
    SEED = int(sys.argv[1])
    np.random.seed(SEED)
    print(choose_100_stocks(SEED))
