=======
rnn_portfolio
=======
---------------------------------------------------------
Stock Portfolio Generation with Recurrent Neural Networks
---------------------------------------------------------

This repository contains tools to produce stock portfolios which
optimize the Sharpe ratio using a recurrent neural net (rnn). 
While the rnn routines can work independently, this package relies
on the `Quantiacs Toolbox`_ backtester for training
and evaluation.

Getting Started
===============

Requirements
------------
Python 2.7 and `pip`_.

Installation
------------
Clone this repository, then use `pip`_ in the repository directory to
install the package::
  
  $ pip install .

This will install all the necessary dependencies.

Tests
-----
To test that your installation was successful, change to the repository
directory and run `pytest`_::

  $ pytest

..
   Usage
   =====

Built With
==========
* `Quantiacs Toolbox`_
* `TensorFlow`_
* `SciPy`_
* `scikit-learn`_
* `pytest`_

Versioning
==========
We use `SemVer`_.

License
=======
This project is licensed under the `MIT license`_.

.. _pip: http://www.pip-installer.org/en/latest/
.. _SemVer: http://semver.org/
.. _pytest: http://doc.pytest.org/en/latest/
.. _Quantiacs Toolbox: https://www.quantiacs.com/For-Quants/GetStarted/QuantiacsToolbox.aspx
.. _SciPy: https://www.scipy.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _TensorFlow: https://www.tensorflow.org/
.. _MIT license: https://opensource.org/licenses/MIT
