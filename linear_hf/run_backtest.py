import sys
import random

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

import neuralnet
from preprocessing import non_nan_markets
from preprocessing import nan_markets
from preprocessing import returns_check
from preprocessing import load_nyse_markets
from preprocessing import preprocess
from batching_splitting import split_validation_training
from costs import compute_numpy_sharpe

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):

    market_data, all_data, should_retrain = preprocess(
        settings['markets'], OPEN, CLOSE, HIGH, LOW, VOL, DATE, 
        CLOSE_LASTTRADE, CLOSE_ASK, CLOSE_BID, RETURN, SHARE, 
        DIVIDEND, TOTALCAP, postipo=100, filler=0.123456789, 
        data_types = settings['data_types'])
    # Calculate Sharpe between training intervals
    recent_cost = calculate_recent(iteration=settings['iter'], 
                                     retrain_interval=settings['retrain_interval'], 
                                     exposure=exposure, 
                                     market_data=market_data,
                                     cost=settings['cost_type'])
    
    print_things(settings['iter'],DATE[-1],fundEquity[-1],settings['val_sharpe'],recent_cost)
    
    if fundEquity[-1] < .75:
        raise ValueError('Strategy lost too much money')

    if settings['iter'] == 0:
        print 'Initializing net...\n'
        # Define a new neural net.
        settings['nn'] = neuralnet.Linear(n_ftrs=all_data.shape[1], 
                                          n_markets=OPEN.shape[1],
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'],
                                          allow_shorting=settings['allow_shorting'],
                                          cost=settings['cost_type'])
        print 'Done with initializing neural net!'

    # Train the neural net on current data.
    best_val_sharpe = -np.inf
    if settings['iter'] % settings['retrain_interval'] == 0:
        best_val_sharpe = -np.inf
        best_tr_loss = np.inf
        if settings['restart_variables']:
            settings['nn'].restart_variables()
                
        batches_per_epoch = calc_batches(all_data.shape[0], settings)
        
        for epoch_id in range(settings['num_epochs']):
            seed = np.random.randint(10000)
            tr_sharpe = 0.
            val_sharpe = 0.
            lr_new = lr_calc(settings, epoch_id)
            for batch_id in range(batches_per_epoch):
                all_val, market_val, all_batch, market_batch = split_validation_training(
                    all_data, market_data, 
                    valid_period=settings['val_period'], 
                    horizon=settings['horizon'], 
                    n_for_sharpe=settings['n_sharpe'],
                    batch_id=batch_id, 
                    batch_size=settings['batch_size'],
                    randseed=seed)

                settings['nn'].train_step(batch_in=all_batch, batch_out=market_batch, lr=lr_new)
                tr_sharpe += loss_calc(settings,all_batch,market_batch)

            if settings['val_period'] > 0:
                val_loss = settings['nn'].loss_np(all_val, market_val)
                val_l1_loss = settings['nn'].l1_penalty_np()
                val_sharpe = -(val_loss - val_l1_loss)
                
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    settings['nn'].save()

                if best_val_sharpe > settings['val_sharpe_threshold']:
                    settings['dont_trade'] = False
                else:
                    settings['dont_trade'] = True
                
                # Record val_sharpe for results
                settings['val_sharpe'] = best_val_sharpe

            elif loss < best_tr_loss:
                best_tr_loss = loss
                settings['nn'].save()

            tr_sharpe /= batches_per_epoch
            sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
                epoch_id, val_sharpe, tr_sharpe))
            sys.stdout.flush()
        
    # Predict a portfolio.
    settings['nn'].load()
    positions = settings['nn'].predict(all_data[-settings['horizon']:])
    if settings['dont_trade']:
        positions = dont_trade_positions(positions, settings)
   
    # Save validation sharpes and actualized sharpes!
    settings['realized_sharpe'].append(recent_cost)
    settings['saved_val_sharpe'].append(best_val_sharpe)
    
    settings['iter'] += 1
    return positions, settings


def mySettings():
    settings={}
    # Futures Contracts
    settings['n_time'] =  40 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 10 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = .1 # L1 regularizer strength.
    settings['num_epochs'] = 30 # Number of epochs each day.
    settings['batch_size'] = 128
    settings['val_period'] = 32
    settings['lr'] = 1e-4 # Learning rate.
    settings['dont_trade'] = False # If on, don't trade.
    settings['iter'] = 0
    settings['lookback'] = 252
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    #settings['beginInSample'] = '20090102'
    #settings['endInSample'] = '20131231'
    settings['beginInSample'] = '20000601'
    settings['endInSample'] = '20140101'
    settings['val_sharpe_threshold'] = -np.inf
    settings['retrain_interval'] = 30
    settings['realized_sharpe'] = []
    settings['saved_val_sharpe'] = []
    settings['val_sharpe'] = -np.inf
    settings['cost_type'] = 'mean_return'
    settings['allow_shorting'] = True
    settings['lr_mult_base'] = .1
    settings['restart_variables'] = True
    ''' Pick data types to feed into neural net. If empty, only CLOSE will be used. 
    Circle dates added automatically if any setting is provided. 
    0 = OPEN
    1 = CLOSE
    2 = HIGH
    3 = LOW
    4 = VOL
    5 = CLOSE_LASTTRADE
    6 = CLOSE_ASK
    7 = CLOSE_BID
    8 = RETURNS
    9 = SHARES
    10 = DIVIDENDS
    11 = TOTALCAPS
    12 = DATE
    '''
    settings['data_types'] = [0]
    
    # Hard code markets for testing.
    #settings['markets'] = ['85653_AI_nyse', '81061_MCK_nyse', '62148_CSX_nyse', '85621_MTD_nyse', '85442_TSM_nyse', '36468_SHW_nyse', '50788_ESL_nyse', '46690_CBT_nyse', '86023_MUS_nyse', '77925_MQT_nyse', '77823_KMP_nyse', '32678_HEI_nyse', '87198_ARLP_nasdaq', '87289_SNH_nyse', '77971_EFII_nasdaq', '77860_JEQ_nyse', '55001_TRN_nyse', '79323_ALL_nyse', '47941_TGNA_nyse', '46463_GFF_nyse', '51633_VVC_nyse', '80691_LPT_nyse', '75429_PHF_nyse_mkt', '77604_ALU_nyse', '22840_HSH_nyse', '75183_PCF_nyse', '19350_DE_nyse', '78034_PDCO_nasdaq', '38762_NI_nyse', '75278_AB_nyse', '49656_BK_nyse', '11369_UBSI_nasdaq', '79665_TEI_nyse', '24328_EQT_nyse', '76697_HNT_nyse', '61778_KYO_nyse', '82924_JW_nyse', '84042_PAG_nyse', '83604_SKM_nyse', '55213_RT_nyse', '86143_VVR_nyse', '70826_MFM_nyse', '85421_CHL_nyse', '59504_BTI_nyse_mkt', '14816_TR_nyse', '79363_AZN_nyse', '68187_WRI_nyse', '73139_SYK_nyse', '86102_FII_nyse', '77078_TOT_nyse', '78775_CREAF_nasdaq', '88249_SCRX_nasdaq', '85074_LQ_nyse', '79195_PERY_nasdaq', '84351_BHBC_nasdaq', '79795_ASCA_nasdaq', '78758_BBHL_nasdaq', '86290_CVV_nasdaq', '89556_MPQ_nyse_mkt', '12236_NOVN_nasdaq', '77420_FRED_nasdaq', '89374_PCO_nyse', '80122_FFLC_nasdaq', '11634_RBNC_nasdaq', '87268_CIR_nyse', '81705_FEIC_nasdaq', '76515_DHC_nyse_mkt', '11292_ICCC_nasdaq', '88784_ADLR_nasdaq', '23799_CIA_nyse', '91265_HAXS_nasdaq', '69892_GG_nyse', '89103_GLAD_nasdaq', '82261_GSIG_nasdaq', '85686_DEPO_nasdaq', '76037_CBC_nyse', '89928_FBTX_nasdaq', '88550_OPNT_nasdaq', '38172_WOC_nyse_mkt', '77555_REM_nyse', '86313_UBA_nyse', '80539_NKTR_nasdaq', '86839_OBAS_nasdaq', '83533_MXIC_nasdaq', '82207_TRKN_nasdaq', '79358_VMV_nyse_mkt', '85510_TONS_nasdaq', '37460_TLX_nyse_mkt', '86915_RMIX_nasdaq', '77262_TTES_nasdaq', '88836_GNVC_nasdaq', '87649_CHRD_nasdaq', '83145_IRIX_nasdaq', '84562_HGRD_nasdaq', '87762_CRYP_nasdaq', '17778_BRK_nyse', '89929_FLCN_nasdaq', '87663_IPK_nyse_mkt', '89495_TAP_nyse', '76691_AETC_nasdaq', 'CASH']
    settings['markets'] = load_nyse_markets(start_date=settings['beginInSample'], 
                                            end_date=settings['endInSample'],
                                            lookback=0,
                                            postipo=0)
    settings['markets'] = settings['markets'][-10:] + ['CASH']
    print(settings['markets'])
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print(results['stats'])


def calculate_recent(iteration, retrain_interval, exposure, market_data, cost='sharpe'):
    """ Calculate the realized sharpe ratios from the output of the neural net
        using the latest trading positions.

    Args:
        iteration: iteration number in backtester, used to calculate n_days_back
        retrain_interval: how often the nn retrains
        exposure: output from the backtester, settings['exposure'] (ntimesteps, nmarkets)
        market_data: output from the backtester, [open close high low] (ntimesteps, nmarkets) 

    Returns:
        Cost function estimate:
            if cost = 'sharpe', returns annualized sharpe since last retrain
            if cost = 'min_return', returns minimium of the daily return since last retrain
            if cost = 'mean_return', returns the mean of the returns since last retrain
            if cost = 'mixed_return', returns mean*min of returns since last retrain
    """

    # Calculate Sharpe between training intervals. iteration-1 is used because we score one day back.
    n_days_back = np.mod(iteration,retrain_interval)
     
    # Only start scoring realized sharpes from greater than 2!
    if n_days_back > 2:
        recent_sharpe = compute_numpy_sharpe(positions=exposure[None, -n_days_back-3:-1, :],
                             prices=market_data[None, -n_days_back-2:, :],
                             slippage=0.05, n_ignore=2)
        recent_returns = compute_numpy_sharpe(positions=exposure[None, -n_days_back-3:-1, :],
                             prices=market_data[None, -n_days_back-2:, :],
                             slippage=0.05, n_ignore=2, return_returns = True)
        
        if np.isnan(recent_sharpe):
            # NaNs out when all positions are cash, therefore std.dev(ret) = 0
            recent_sharpe = 0
    else:
        # Return nans for the first couple days.
        recent_sharpe = np.nan
        recent_returns = np.array([np.nan, np.nan])

    if cost == 'sharpe':        
        return recent_sharpe
    elif cost == 'min_return':
        return recent_returns.min()
        return returns.mean() * returns.min()
    elif cost == 'mean_return':
        return recent_returns.mean()
    elif cost == 'mixed_return':
        return recent_returns.mean() * recent_returns.min()


def print_things(iteration,DATE,fundEquity,val_sharpe,recent_cost):
    """ Prints out things

    Args:
        iteration: iteration in backtester
        DATE: last date
        fundEquity: last fundEquity amount
        val_sharpe: most recent validation sharpe saved in settings
        recent_cost: most recent recent_cost.
    """
    print('Iter {} [{}], equity {}.'.format(iteration, 
                                            DATE,
                                            fundEquity))
    if iteration > 1:
        print('[Recent validation costfn] Recent costfn: [{}] {}'.format(
                                            val_sharpe,
                                            recent_cost))


def dont_trade_positions(positions, settings):
    """ Sets positions to zero, and cash to 1

    Args:
        positions: positions output from neuralnet
        settings: needed to find 'CASH' index
    Returns:
        positions: all zeroed out except for CASH, which is 1
    """
    
    positions *= 0 
    cash_index = settings['markets'].index('CASH')
    positions[cash_index] = 1
    return positions


def calc_batches(n_timesteps, settings):
    """ Calculates the total groups of batch_size that are one epoch.
    
    Args:
        n_timesteps: total number of timesteps
        settings: takes horizon, val_period, n_sharpe, and batch_size
    Returns:
        batches_per_epoch: the floor of total_possible_timesteps/batch_size,
                           where total_possible is taken from batches
    """
    
    batches_per_epoch = int(np.floor((n_timesteps -
                                          settings['horizon'] -
                                          settings['val_period'] -
                                          2 * settings['n_sharpe'] + 1)
                                         /float(settings['batch_size'])))
    return batches_per_epoch


def lr_calc(settings, epoch_id)
    """ Learning rate update based on epoch_id

    Args:
        settings: used to caluclate an lr multiplier
        epoch_id: used to calculate the new lr
    Return:
        lr_new: new learning rate, dependent on epoch_id, lr, lr_mult_base.
    """
    lr_mult = settings['lr_mult_base'] ** (1. / settings['num_epochs'])
    lr_new = settings['lr'] * lr_mult ** epoch_id

    return lr_new

def loss_calc(settings, all_batch, market_batch)
    """ Calculates loss from neuralnet

    Args: 
        settings: contains the neural net
        all_batch: the inputs to neural net
        market_batch: [open close high low] used to calculate loss
    Returns:
        cost: loss - l1 penalty
    """
    loss = settings['nn'].loss_np(all_batch, market_batch)
    l1_loss = settings['nn'].l1_penalty_np()
    return -(loss - l1_loss)

