=======
CauseHF
=======
---------------------------------------------------------
Stock Portfolio Generation with Recurrent Neural Networks
---------------------------------------------------------

This repository contains tools to produce stock portfolios which
optimize various measures (such as the Sharpe ratio) using recurrent
neural networks.

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

To do:
======
* Clean code.
* Add fake data.
* Add to package_data in setup.py.
* Improve README.
* Remove non-recurrent neural net.
* Put on GitHub.
* Consider speeding up tests.

.. _pip: http://www.pip-installer.org/en/latest/
.. _SemVer: http://semver.org/
.. _pytest: http://doc.pytest.org/en/latest/
.. _Quantiacs Toolbox: https://www.quantiacs.com/For-Quants/GetStarted/QuantiacsToolbox.aspx
.. _SciPy: https://www.scipy.org/
.. _scikit-learn: http://scikit-learn.org/stable/
.. _TensorFlow: https://www.tensorflow.org/
.. _MIT license: https://opensource.org/licenses/MIT
