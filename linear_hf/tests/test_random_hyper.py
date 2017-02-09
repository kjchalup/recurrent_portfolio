"""Test randomly supplied hyper parameters."""
from linear_hf.random_hyper import supply

def test_supply():
    """Supply should take a dictionary of keys for the settings and
inclusive ranges of numbers and return a random parameter from inside the range.
"""

    # Make fake ranges.
    setrange = {'this': [2, 9], 'that': [0, 42], 'when': [-30, 12]}

    # Get settings.
    settings = supply(setrange)

    # Test that keys match.
    assert settings.keys() == setrange.keys()

    # Test that values are in range.
    for setting in settings:
        low = setrange[setting][0]
        high = setrange[setting][1]
        print(setting)
        print(settings[setting])
        assert settings[setting] in range(low, high+1)
