"""Test randomly supplied hyper parameters."""
from linear_hf.hyperparameters import supply_hypers
from linear_hf.hyperparameters import CHOICES, N_SHARPE_MIN, N_SHARPE_GAP

def test_hypers():
    """Test the supply_hypers function."""

    settings = supply_hypers()
    CHOICES['n_sharpe'] = range(N_SHARPE_MIN, settings['n_time'] - N_SHARPE_GAP)

    # Test that keys match.
    assert set(settings.keys()) == set(CHOICES.keys())

    # Test that values are in range.
    for setting in settings:
        print(settings)
        assert settings[setting] in CHOICES[setting]
