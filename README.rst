.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

*Stock Portfolio Generation and Backtesting with Recurrent Neural Networks*

Usage
-----
All the setup boils down to modifying the `settings` dictionary in `run_backtest.py`_:

.. code:: python
  settings['markets']
  settings['beginInSample'] = 20010101
  settings['lookback'] = 252

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

  $ py.test

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
