"""Add runslow flag for test functions."""
import pytest

def pytest_addoption(parser):
    """Add runslow and runfast command line options.

    To use, add the following to your module:

slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

    and then decorate the slow tests, like so:
@slow
def test_slow():
    assert 2 == 2

    Replace slow with fast if appropriate.
"""
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")
    parser.addoption("--runfast", action="store_true",
                     help="run fast tests")
