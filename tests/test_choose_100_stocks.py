""" Test for choose_100_stocks.py. """
import joblib
from linear_hf.choose_100_stocks import choose_100_stocks

def test_choose_100_stocks():
    """ Test for choose_100_stocks.py. """
    stocks = choose_100_stocks(1)
    otherstocks = choose_100_stocks(3)
    assert len(stocks) == 100
    assert len(set(stocks)) == 100
    assert set(otherstocks) != set(stocks)

    allstocks = joblib.load('linear_hf/1000_stock_names.pkl')
    for stock in stocks:
        if stock != 'CASH':
            assert stock in allstocks
