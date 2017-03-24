.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*Stock Portfolio Generation and Backtesting with Recurrent Neural Networks. Maintained by Krzysztof Chalupka, Tristan McKinney, Alex Teng.*

Usage
-----
All the setup boils down to modifying the `mySettings()` function in `run_backtest.py`_:

.. code:: python 

    def mySettings():
        """ Settings for the backtester"""
        settings = {}
        settings['n_time'] = 252  # Sequence length for the rnn.
        settings['n_sharpe'] = 252  # Optimize Sharpe for this many timesteps.
        settings['num_epochs'] = 100  # Train for this many epochs.
        settings['batch_size'] = 32  # SGD batch size.
        settings['val_period'] = 16  # Validation period, taken off the most recent timesteps.
        settings['lr'] = 1e-2  # Learning rate.
        settings['lookback'] = 1000  # On first day of trading, use this many past days for training.
        settings['beginInSample'] = '20010104'  # First day of data.
        settings['endInSample'] = '20131231'  # Stop trading here.
        settings['retrain_interval'] = 100  # Retrain the rnn every this many trading days.
        settings['allow_shorting'] = True  # Whether to allow the rnn to keep short positions.
        settings['restart_variables'] = True  # Whether to retrain the rnn from scratch each time.
        settings['data_types'] = [1, 4]  # Input data for rnn (OPEN, CLOSE, HIGH, LOW, DATE = 0, ..., 4).
        settings['markets'] = ['AAPL', 'GOOG', 'MMM', 'CASH']  # Which markets to trade.
        # [more settings follow, not shown in this readme]

Comments in the file should explain what these settings do. Importantly, note that if
`settings['markets']` contains symbols available through `Quantiacs Toolbox`_, the
data will be automatically downloaded into the tickerData directory. You can, however,
use your own data. We got ours from `CRSP`_ because `Quantiacs Toolbox`_ does not
offer survivorship-bias free data. You'll need to format your ticker data into the same
format as Quantiacs data, and put it in the tickerData directory.

Once done choosing the settings, run

    $ python -m rnn_portfolio.run_backtest
    
This will start the backtest run. Each `settings['retrain_interval']` days, the
rnn will retrain using all the past data. The rnn optimizes Sharpe ratios over
`settings['n_sharpe']`-day periods in the training data. Its output is the portfolio
for the next day. At the end of the run, you should see a nice plot (made by `Quantiacs Toolbox`_)
showing how much you earned or lost:

Installation
------------
Use Python 2.7 and an up-to-date `pip`_ for easy installation.
Our rnn code is  `Tensorflow`_-based. If you can, run it on a
machine with a GPU. Clone this repository, then install the package
from within the repository directory::
  
  $ pip install .

This will install all the necessary dependencies. To test that your
installation was successful, change to the repository directory
and run `pytest`_::

  $ pytest

.. _CRSP: http://www.crsp.com/
.. _run_backtest.py: rnn_portfolio/run_backtest.py
.. _pip: http://www.pip-installer.org/en/latest/
.. _SemVer: http://semver.org/
.. _pytest: http://doc.pytest.org/en/latest/
.. _Quantiacs Toolbox: https://www.quantiacs.com/For-Quants/GetStarted/QuantiacsToolbox.aspx
.. _SciPy: https://www.scipy.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _TensorFlow: https://www.tensorflow.org/
.. _MIT license: https://opensource.org/licenses/MIT
.. _run_backtest.py: rnn_portfolio/run_backtest.py
