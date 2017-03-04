""" Functions that load data from txt files, as well as clean
loaded data of NaNs, zeros and other oddities.

"""
import glob
from  datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np
from . import NP_DTYPE

def load_nyse_markets(start_date, postipo=100):
    """ Loads nyse markets which start before start_date-postipo.

    Args:
        start_date: date of starting to consider data
        postipo: number of days stock befor start_date the stock must
        start to be considered.

    """
    all_nyse = glob.glob('tickerData/*nyse.txt')
    alives = []

    start_date_minuspostipo = (datetime.strptime(start_date, '%Y%m%d') -
                               timedelta(days=postipo)).strftime('%Y%m%d')
    for fname in all_nyse:
        data = open(fname, 'r').readlines()
        if (int(data[1].split(',')[0]) < int(start_date_minuspostipo) and
                int(data[-1].split(',')[0]) > int(start_date)):
            alives.append(fname)
    assert len(alives) > 0, "No stocks returned! Check start_date-postipo is OK!"
    return [symbol.split('/')[1][:-4] for symbol in alives]


def preprocess(markets, opens, closes, highs, lows, vols, dates,
               close_lasttrade, close_ask, close_bid, returns, shares,
               dividends, totalcaps, postipo=100, filler=0.0000001,
               data_types=[]):
    """Preprocesses stock price data for use in our neural nets.

    Replaces nans in market price data imported using quantiacsToolbox.py using
    the following rules:
    1) If the first day of closing prices contains nans, it is assumed that
    the stock has not yet been listed on an index.
    2) For such a stock, the open, close, high, and low prices are replaced
    with a filler value if the last day of provided stock data is less than a
    specified number of days (given by the parameter postipo) after the final
    nan in the closing prices.
    3) If the last day of provided stock data is more than postipo days after
    the final nan in the closing prices, the initial nans are replaced with the
    closing price on the first non-nan day.
    4) The first non-nan day's open, high, and low prices are also replaced
    with the close on that day.
    5) Any other nans in the closing prices are filled with the previous valid
    closing price. This includes both delisted stocks and nans from errors in
    the initial data.
    6) After these steps, any remaining nans in opens, highs, and lows are
    replaced with the closing price on the same day.
    7) The closing prices of the market CASH are replaced by a constant value.
    8) All of the input data is normalized to make their values close to 1,
    to improve the performance of the neural net training.
    9) Any nans in other price data are replaced by zeros.
    10) Selects correct data for all_data from data_types.

    Args:
        markets (list): names of markets
        opens, ..., totalcaps (np arrays): market prices given stock index
            and day index, indices look like opens[day][stock]
        dates (np array): dates in yyyymmdd format
        postipo (int): number of days to wait to begin including prices
        filler (float): value to fill with
        data_types (list): list of selected features.

    Returns:
        filled_prices (np array): horizontally concatenated array of
            preprocessed o, c, h, l
        all_data (np array): horizontally concatenated array of all data
        should_retrain (np array): boolean array, length number of stocks,
            which indicates which stocks were postipo days after their initial
            non-nan closing price on the final day of price data
    """
    # Normalize values by magic numbers to stay within reasonable ranges.
    n_markets = opens.shape[1]
    divide_prices_by = 50000.
    opens = opens / divide_prices_by
    closes = closes / divide_prices_by
    highs = highs / divide_prices_by
    lows = lows / divide_prices_by
    close_lasttrade = close_lasttrade / divide_prices_by
    close_ask = close_ask / divide_prices_by
    close_bid = close_bid / divide_prices_by
    dividends = dividends / divide_prices_by
    divide_vol_by = 1000000.
    vols = vols / divide_vol_by
    divide_tcap_by = 3000000.
    totalcaps = totalcaps / divide_tcap_by
    divide_shares_by = 100000.
    shares = shares / divide_shares_by
    returns[returns < -1] = 0.
    # Make list of stocks for which close starts as nan. We will assume these
    # are preipo stocks in the data.
    cnans = np.isnan(closes)
    preipo = cnans[0]

    # Copy prices to make sure not to clobber past prices when nanning things.
    closes_copy = np.array(closes)

    # Prices other than closes.
    prices = [opens, highs, lows]
    prices_copy = []
    for price in prices:
        prices_copy.append(np.array(price))

    # Compute the number of days after nans stop for a particular stock in close.
    daysipo = np.logical_not(cnans).cumsum(0)

    # Loop throught the days in closes.
    last_close = closes[0]
    if 'CASH' in markets:
        cashindex = markets.index('CASH')

    for day, close in enumerate(closes):
        # Replace nans with previous close in closes and closes_copy.
        closes_copy[day, np.isnan(close)] = last_close[np.isnan(close)]
        close[np.isnan(close)] = last_close[np.isnan(close)]

        # Replace closes which don't have enough days after ipo with nans.
        tonan = np.logical_and(daysipo[day] < postipo, preipo)
        close[tonan] = np.nan

        # Do the same for the other prices.
        for price in prices:
            price[day, tonan] = np.nan

        # If enough days have passed since ipo, replace old nans with first
        # non-nan closing price.
        if day >= postipo:
            enoughdays = daysipo[day] == postipo
            closes[:day, enoughdays] = np.vstack(
                (np.tile(closes_copy[day-postipo+1, enoughdays],
                         (day-postipo+1, 1)),
                 closes_copy[day-postipo+1:day, enoughdays]))

            # And for the other prices, replace old with first non-nan close,
            # but restore the infomation about the other prices except on the
            # first non-nan day, where we will replace everything with close.
            for count, price in enumerate(prices):
                price[:day+1, enoughdays] = np.vstack(
                    (np.tile(closes_copy[day-postipo+1, enoughdays],
                             (day-postipo+2, 1)),
                     prices_copy[count][day-postipo+2:day+1, enoughdays])
                )
        else:
            enoughdays = np.zeros((len(close)), dtype=bool)

        if 'CASH' in markets:
            close[cashindex] = 1/divide_prices_by

        last_close = close

    # The last value of enoughdays will tell us whether we "turned on" a stock.
    should_retrain = enoughdays

    # Fill remaining nans in close with filler. These should only be stocks
    # which have not had enough days since ipo.
    closes[np.isnan(closes)] = filler

    # Fill all remaining nans in price matrices with closes.
    for price in prices:
        price[np.isnan(price)] = closes[np.isnan(price)]

    # Construct price matrix to return.
    filled_prices = np.hstack((opens, closes, highs, lows))

    # Turn dates into a unit circle.
    y_date, x_date = circle_dates(dates)
    all_data = np.hstack((opens, closes, highs, lows, vols, close_lasttrade,
                          close_ask, close_bid, returns, shares, dividends,
                          totalcaps, x_date[:, None], y_date[:, None]))
    all_data = all_data.astype(NP_DTYPE)
    all_data[np.isnan(all_data)] = 0

    # Run backtester with preprocessing.
    if len(data_types) == 0:
        # If no data_types are chosen, uses standard scaler on OPEN data.
        all_data = StandardScaler().fit_transform(all_data[:, :n_markets])
    else:
        # Otherwise select the datatypes required.
        data = np.hstack([all_data[:, n_markets * j: n_markets * (j + 1)]
                          for j in data_types])
        all_data = data

    # Returns check to make sure nothing crazy happens!
    assert np.isnan(filled_prices).sum() == 0
    assert np.isinf(filled_prices).sum() == 0
    assert np.isnan(all_data).sum() == 0
    assert np.isinf(all_data).sum() == 0
    return filled_prices, all_data, should_retrain


def circle_dates(dates):
    '''
    Transform the dates into a unit circle so the algos can learn about
    seasonality. Takes a date of the form 20161231, calculates the equivalent
    (x,y) coordinate using sine and cosine.

    Args:
        dates: list of dates specified as %Y%m%d

    Returns:
        x_date, y_date: unit circle of dates for a year with 366 days
    '''

    # The days in each month of the year.
    month_vec = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    y_date = np.zeros(dates.shape[0])
    x_date = np.zeros(dates.shape[0])
    # For each date, calculate the total days in the months preceding,
    # add to curret day of month.
    for ind, date in enumerate(dates.flatten()):
        month = int(str(date)[4:6])
        if month > 1:
            month_days = (month_vec[0:month-1]).sum()
        else:
            month_days = 0
        day = int(str(date)[6:8])

        # Using total days, divide by 366 to turn into approximate fractional
        # year.
        frac_of_year = (month_days+day)/float(366)

        # Convert the fractional year into radians, take the sine/cosine.
        y_date[ind] = np.sin(frac_of_year*2*np.pi)
        x_date[ind] = np.cos(frac_of_year*2*np.pi)
    return y_date, x_date


def fillnans(data):
    ''' Fill in (column-wise) value gaps with the most recent non-nan value.

    Leading nans remain in place. The gaps are filled-in
    only after the first non-nan entry.

    Args:
        data (iterable)

    Returns:
        an array of the same size as data with the
        nan-values replaced by the most recent non-nan entry.
    '''

    data = np.array(data, dtype=float)
    for row, col in zip(np.where(np.isnan(data))):
        if row > 0:
            data[row, col] = data[row - 1, col]
    return data
