"""Determine the optimal hyperparameters for the neural net."""
import sys
import random

import numpy as np
import joblib

import neuralnet
from preprocessing import non_nan_markets
from preprocessing import nan_markets
from preprocessing import returns_check
from preprocessing import preprocess
from batching_splitting import split_validation_training
from costs import compute_numpy_sharpe

import itertools

from run_backtest import myTradingSystem

def powerset(iterable):
    """ Returns the set of all subsets of the iterable.

    Args:
      iterable: A Python iterable.

    Returns:
      result: The set of all subsets of the iterable, including the empty set.
    """
    all_vals = list(iterable)

    result = list(itertools.chain.from_iterable(
        itertools.combinations(all_vals, this_val)
        for this_val in range(len(all_vals)+1)))
    return result

# Define constants for use in choosing hyperparameters.
LBDS = np.append(10.**np.arange(-5, 3), [0.])
CHOICES = {'n_time': range(21, 253), # Timesteps in one datapoint.
           'lbd': LBDS,              # L1 regularizer strength.
           'num_epochs': range(1, 51),   # Number of epochs each day.
           'batch_size': [32, 64, 128],  # Batch size.
           'lr': 10.**np.arange(-5, 0),  # Learning rate.
           'allow_shorting': [True, False],
           'lookback' : [200, 300, 400, 500, 600, 700, 800, 900, 1000],
           'val_period' : [0, 2, 4, 8, 16, 32],
           'val_sharpe_threshold' : [-np.inf, 0, 1, 2],
           'retrain_interval' : [1] + range(10, 252),
           'data_types' : [[1] + list(j) for j in powerset([0] + range(2, 13))]}

N_SHARPE_MIN = 10 # Minimum value for n_sharpe.
N_SHARPE_GAP = 10 # n_sharpe's max is this much less than n_time.
N_RUNS = 2 # ??? - KC

def mySettings(): # pylint: disable=invalid-name,too-many-arguments
    """ Settings for strategy. """
    settings = {}
    settings = joblib.load('saved_data/hypers.pkl')

    # Only keep markets that have not died out by beginInSample.
    np.random.seed(1)
    random.seed(1)
    settings['markets'] = ['85653_AI_nyse', '81061_MCK_nyse', '62148_CSX_nyse', '85621_MTD_nyse', '85442_TSM_nyse', '36468_SHW_nyse', '50788_ESL_nyse', '46690_CBT_nyse', '86023_MUS_nyse', '77925_MQT_nyse', '77823_KMP_nyse', '32678_HEI_nyse', '87198_ARLP_nasdaq', '87289_SNH_nyse', '77971_EFII_nasdaq', '77860_JEQ_nyse', '55001_TRN_nyse', '79323_ALL_nyse', '47941_TGNA_nyse', '46463_GFF_nyse', '51633_VVC_nyse', '80691_LPT_nyse', '75429_PHF_nyse_mkt', '77604_ALU_nyse', '22840_HSH_nyse', '75183_PCF_nyse', '19350_DE_nyse', '78034_PDCO_nasdaq', '38762_NI_nyse', '75278_AB_nyse', '49656_BK_nyse', '11369_UBSI_nasdaq', '79665_TEI_nyse', '24328_EQT_nyse', '76697_HNT_nyse', '61778_KYO_nyse', '82924_JW_nyse', '84042_PAG_nyse', '83604_SKM_nyse', '55213_RT_nyse', '86143_VVR_nyse', '70826_MFM_nyse', '85421_CHL_nyse', '59504_BTI_nyse_mkt', '14816_TR_nyse', '79363_AZN_nyse', '68187_WRI_nyse', '73139_SYK_nyse', '86102_FII_nyse', '77078_TOT_nyse', '78775_CREAF_nasdaq', '88249_SCRX_nasdaq', '85074_LQ_nyse', '79195_PERY_nasdaq', '84351_BHBC_nasdaq', '79795_ASCA_nasdaq', '78758_BBHL_nasdaq', '86290_CVV_nasdaq', '89556_MPQ_nyse_mkt', '12236_NOVN_nasdaq', '77420_FRED_nasdaq', '89374_PCO_nyse', '80122_FFLC_nasdaq', '11634_RBNC_nasdaq', '87268_CIR_nyse', '81705_FEIC_nasdaq', '76515_DHC_nyse_mkt', '11292_ICCC_nasdaq', '88784_ADLR_nasdaq', '23799_CIA_nyse', '91265_HAXS_nasdaq', '69892_GG_nyse', '89103_GLAD_nasdaq', '82261_GSIG_nasdaq', '85686_DEPO_nasdaq', '76037_CBC_nyse', '89928_FBTX_nasdaq', '88550_OPNT_nasdaq', '38172_WOC_nyse_mkt', '77555_REM_nyse', '86313_UBA_nyse', '80539_NKTR_nasdaq', '86839_OBAS_nasdaq', '83533_MXIC_nasdaq', '82207_TRKN_nasdaq', '79358_VMV_nyse_mkt', '85510_TONS_nasdaq', '37460_TLX_nyse_mkt', '86915_RMIX_nasdaq', '77262_TTES_nasdaq', '88836_GNVC_nasdaq', '87649_CHRD_nasdaq', '83145_IRIX_nasdaq', '84562_HGRD_nasdaq', '87762_CRYP_nasdaq', '17778_BRK_nyse', '89929_FLCN_nasdaq', '87663_IPK_nyse_mkt', '89495_TAP_nyse', '76691_AETC_nasdaq'][:10] + ['CASH']
    return settings

def supply_hypers():
    """Supply hyperparameters to optimize the neural net."""

    np.random.seed()
    random.seed()

    # Get random choices from the ranges (inclusive).
    settings = {}
    for setting in CHOICES.keys():
        settings[setting] = random.choice(CHOICES[setting])

    # Get n_sharpe using n_time.
    settings['n_sharpe'] = random.randint(
        N_SHARPE_MIN, settings['n_time'] - N_SHARPE_GAP)

    return settings

if __name__ == '__main__':
    HYPER_RESULTS = []
    for run in range(N_RUNS):
        # Get hyperparameters.
        SETTINGS = supply_hypers()

        # Other SETTINGS.
        SETTINGS['horizon'] = SETTINGS['n_time'] - SETTINGS['n_sharpe'] + 1
        SETTINGS['iter'] = 0
        SETTINGS['lookback'] = 1000
        SETTINGS['budget'] = 10**6
        SETTINGS['slippage'] = 0.05
        SETTINGS['val_period'] = 0
        SETTINGS['beginInSample'] = '20080102'
        SETTINGS['endInSample'] = '20131201'

        # Save settings for use in test.
        joblib.dump(SETTINGS, 'saved_data/hypers.pkl')

        # Run the strategy.
        import quantiacsToolbox
        RESULTS = quantiacsToolbox.runts(__file__, plotEquity=False)

        # Show the results.
        RESULTS['settings']['nn'] = None
        print [str(hyper) +': ' + str(SETTINGS[hyper])
               for hyper in SETTINGS and CHOICES]
        print ['n_time: ' + str(SETTINGS['n_time'])]
        print RESULTS['stats']

        # Reduce the size of the results files.
        HYPER_RESULTS.append(RESULTS)

        # Save the results
        joblib.dump(HYPER_RESULTS, 'saved_data/hyper_results.pkl')
