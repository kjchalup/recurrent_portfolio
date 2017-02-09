"""Add runslow flag for test functions."""
import pytest

def pytest_addoption(parser):
    """Add runslow command line option."""
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")
