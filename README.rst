=======
rnn_portfolio
=======
This package is co-authored by Tristan McKinney and Alex Teng.

---------------------------------------------------------
Stock Portfolio Generation and Backtesting with Recurrent Neural Networks
---------------------------------------------------------

Tools to produce stock portfolios which optimize the Sharpe 
ratio using a recurrent neural net (rnn). While the rnn routines
can work independently, this package relies on the `Quantiacs Toolbox`_
backtester for training and evaluation.

Getting Started
===============
Our rnn code is `Tensorflow`_-based. If you can, run it on a machine
with a GPU. It will be much faster, especially if you train on thou-
sands of stock symbols!

Requirements
------------
Python 2.7 and `pip`_.

Installation
------------
You'll need Python 2.7 and an up-to-date `pip`_.
Clone this repository, then install the package
from within the repository directory::
  
  $ pip install .

This will install all the necessary dependencies (and their dependencies):

* `Quantiacs Toolbox`_
* `TensorFlow`_
* `SciPy`_
* `scikit-learn`_
* `pytest`_

Tests
-----
To test that your installation was successful, change to the repository
directory and run `pytest`_::

  $ py.test

Example usage
=============
Our system can learn to trade any stocks, as long as you put their ticker 
files in the tickerData directory in an appropriate format. Let's first run
through a simple example.

Trading with Quantiacs data
---------------------------
Say you're interested in trading only 3M (MMM), Coca-Cola (KO) or keeping cash.
For backtesting purposes, lets travel back in time to 2001, by adjusting the 
`settings` dictionary in `run_backtest.py`_:

.. code:: python
  settings['beginInSample'] = 20010101
  settings['lookback'] = 252


License and versioning
======================
This project is licensed under the `MIT license`_. We use `SemVer`_ for versioning.

.. _pip: http://www.pip-installer.org/en/latest/
.. _SemVer: http://semver.org/
.. _pytest: http://doc.pytest.org/en/latest/
.. _Quantiacs Toolbox: https://www.quantiacs.com/For-Quants/GetStarted/QuantiacsToolbox.aspx
.. _SciPy: https://www.scipy.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _TensorFlow: https://www.tensorflow.org/
.. _MIT license: https://opensource.org/licenses/MIT
.. _run_backtest.py: rnn_portfolio/run_backtest.py
