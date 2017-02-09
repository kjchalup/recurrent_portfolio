"""Supply hyperparameters for the neural net to test."""
from random import randint

def supply(setrange):
    """Take a dictionary of settings and ranges of parameters and return a
random selection from those ranges.

    Args:
        setrange (dict): Dictionary with setting names as keys and integer
            ranges for the (inclusive) range of parameters to choose from.

    Returns:
        settings (dict): Dictionary with settings and randomly chosen
            parameters.
    """

    settings = {}
    for setting, prange in setrange.items():
        settings[setting] = randint(prange[0], prange[1])

    return settings
