import sys
import random

from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

import neuralnet
from preprocessing import non_nan_markets
from preprocessing import nan_markets
from preprocessing import returns_check
from preprocessing import preprocess
from batching_splitting import split_validation_training

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    
    n_markets=len(settings['markets'])
    # Returns check to make sure nothing crazy happens!
    returns_check(OPEN, CLOSE, HIGH, LOW, DATE, settings['markets'])

    market_data, all_data, should_retrain = preprocess(
                                                        settings['markets'], OPEN, CLOSE, HIGH, LOW,
                                                        VOL, DATE, CLOSE_LASTTRADE, CLOSE_ASK, 
                                                        CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP,
                                                        postipo=100, filler=0.123456789)
    # Returns check afte preprocessing to make sure nothing crazy happens!
    returns_check(market_data[:,:n_markets],
                    market_data[:,n_markets:n_markets*2],
                    market_data[:,n_markets*2:n_markets*3],
                    market_data[:,n_markets*3:n_markets*4],
                    DATE, settings['markets'])
    
    assert np.isnan(market_data).sum() == 0
    assert np.isinf(market_data).sum() == 0
    assert np.isnan(all_data).sum() == 0
    assert np.isinf(all_data).sum() == 0
    
    # Run backtester without preprocessing
    #market_data = np.hstack([OPEN, CLOSE, HIGH, LOW])
    #all_data = StandardScaler().fit_transform(OPEN)#np.hstack([OPEN, VOL, DIVIDEND, TOTALCAP]))
    
    # Run backtester with preprocessing
    if len(settings['data_types']) == 0:
        all_data = StandardScaler().fit_transform(all_data[:,:n_markets])
    else:
        data = []
        for j in settings['data_types']:
            data.append(all_data[:,n_markets*(j):n_markets*(j+1)])
        # Append x_date, y_date
        data.append(all_data[:,-2:])
    
        # Stacks chosen data back into correct shape!
        all_data = np.hstack(data)
    
    print('Iter {} [{}], equity {}.'.format(settings['iter'], 
                                            DATE[-1],
                                            fundEquity[-2]))
    #import pdb;pdb.set_trace()
    if settings['iter'] == 0:
        print 'Initializing net...\n'
        # Define a new neural net.
        settings['nn'] = neuralnet.Linear(n_ftrs=all_data.shape[1], 
                                          n_markets=OPEN.shape[1],
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'],
                                          allow_shorting=True)
        print 'Done with initializing neural net!'
    # Train the neural net on current data.
    if settings['iter'] % settings['n_sharpe'] == 0:
        settings['nn'].restart_variables()
        lr_mult = .01 ** (1. / settings['num_epochs'])
        batches_per_epoch = int(np.floor((all_data.shape[0] -
                                          settings['horizon'] -
                                          settings['val_period'] -
                                          2 * settings['n_sharpe'] + 1)
                                         /float(settings['batch_size'])))

        for epoch_id in range(settings['num_epochs']):
            seed = np.random.randint(10000)
            tr_sharpe = 0.
            val_sharpe = 0.
            for batch_id in range(batches_per_epoch):
                all_val, market_val, all_batch, market_batch = split_validation_training(
                    all_data, market_data, 
                    valid_period=settings['val_period'], 
                    horizon=settings['horizon'], 
                    n_for_sharpe=settings['n_sharpe'],
                    batch_id=batch_id, 
                    batch_size=settings['batch_size'],
                    randseed=seed)

                settings['nn'].train_step(
                    batch_in=all_batch, 
                    batch_out=market_batch, 
                    lr=settings['lr'] * lr_mult ** epoch_id)
                loss = settings['nn'].loss_np(all_batch, market_batch)
                l1_loss = settings['nn'].l1_penalty_np()
                tr_sharpe += -(loss - l1_loss)
            if settings['val_period'] > 0:
                val_loss = settings['nn'].loss_np(all_val, market_val)
                val_l1_loss = settings['nn'].l1_penalty_np()
                val_sharpe = -(val_loss - val_l1_loss)
                #import pdb;pdb.set_trace()
                if val_sharpe > 1:
                    settings['dont_trade'] = False
                else:
                    settings['dont_trade'] = False

            tr_sharpe /= batches_per_epoch
            sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
                epoch_id, val_sharpe, tr_sharpe))
            sys.stdout.flush()

    # Predict a portfolio.
    positions = settings['nn'].predict(all_data[-settings['horizon']:])
    #import pdb;pdb.set_trace()
    if settings['dont_trade']:
        positions *= 0 
        cash_index = settings['markets'].index('CASH')
        positions[cash_index] = 1
    settings['iter'] += 1
    return positions, settings


def mySettings():
    settings={}
    # Futures Contracts
    settings['n_time'] =  160 # Use this many timesteps in one datapoint.
    settings['n_sharpe'] = 100 # This many timesteps to compute Sharpes.
    settings['horizon'] = settings['n_time'] - settings['n_sharpe'] + 1
    settings['lbd'] = .0001 # L1 regularizer strength.
    settings['num_epochs'] = 10 # Number of epochs each day.
    settings['batch_size'] = 128
    settings['val_period'] = 1
    settings['lr'] = 1e-3 # Learning rate.
    settings['dont_trade'] = False # If on, don't trade.
    settings['iter'] = 0
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    #settings['beginInSample'] = '20090102'
    #settings['endInSample'] = '20131231'
    settings['beginInSample'] = '20040101'
    settings['endInSample'] = '20140101'

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
    '''
    settings['data_types'] = []
    settings['data_types'] = np.sort(settings['data_types'])
    
    
    # Only keep markets that have not died out by beginInSample.
    np.random.seed(1)
    random.seed(1)
    #import pdb;pdb.set_trace()
    '''
    no_nan_names  = non_nan_markets(settings['beginInSample'], 
                                           settings['endInSample'], 
                                           lookback=settings['lookback'])
    nan_names = nan_markets(settings['beginInSample'],
                                      settings['endInSample'],
                                      lookback=settings['lookback'])
    settings['markets'] = no_nan_names[0:50] + nan_names[0:50] + ['CASH']
    '''
    
    # Hard code markets for testing.
    settings['markets'] = ['85653_AI_nyse', '81061_MCK_nyse', '62148_CSX_nyse', '85621_MTD_nyse', '85442_TSM_nyse', '36468_SHW_nyse', '50788_ESL_nyse', '46690_CBT_nyse', '86023_MUS_nyse', '77925_MQT_nyse', '77823_KMP_nyse', '32678_HEI_nyse', '87198_ARLP_nasdaq', '87289_SNH_nyse', '77971_EFII_nasdaq', '77860_JEQ_nyse', '55001_TRN_nyse', '79323_ALL_nyse', '47941_TGNA_nyse', '46463_GFF_nyse', '51633_VVC_nyse', '80691_LPT_nyse', '75429_PHF_nyse_mkt', '77604_ALU_nyse', '22840_HSH_nyse', '75183_PCF_nyse', '19350_DE_nyse', '78034_PDCO_nasdaq', '38762_NI_nyse', '75278_AB_nyse', '49656_BK_nyse', '11369_UBSI_nasdaq', '79665_TEI_nyse', '24328_EQT_nyse', '76697_HNT_nyse', '61778_KYO_nyse', '82924_JW_nyse', '84042_PAG_nyse', '83604_SKM_nyse', '55213_RT_nyse', '86143_VVR_nyse', '70826_MFM_nyse', '85421_CHL_nyse', '59504_BTI_nyse_mkt', '14816_TR_nyse', '79363_AZN_nyse', '68187_WRI_nyse', '73139_SYK_nyse', '86102_FII_nyse', '77078_TOT_nyse', '78775_CREAF_nasdaq', '88249_SCRX_nasdaq', '85074_LQ_nyse', '79195_PERY_nasdaq', '84351_BHBC_nasdaq', '79795_ASCA_nasdaq', '78758_BBHL_nasdaq', '86290_CVV_nasdaq', '89556_MPQ_nyse_mkt', '12236_NOVN_nasdaq', '77420_FRED_nasdaq', '89374_PCO_nyse', '80122_FFLC_nasdaq', '11634_RBNC_nasdaq', '87268_CIR_nyse', '81705_FEIC_nasdaq', '76515_DHC_nyse_mkt', '11292_ICCC_nasdaq', '88784_ADLR_nasdaq', '23799_CIA_nyse', '91265_HAXS_nasdaq', '69892_GG_nyse', '89103_GLAD_nasdaq', '82261_GSIG_nasdaq', '85686_DEPO_nasdaq', '76037_CBC_nyse', '89928_FBTX_nasdaq', '88550_OPNT_nasdaq', '38172_WOC_nyse_mkt', '77555_REM_nyse', '86313_UBA_nyse', '80539_NKTR_nasdaq', '86839_OBAS_nasdaq', '83533_MXIC_nasdaq', '82207_TRKN_nasdaq', '79358_VMV_nyse_mkt', '85510_TONS_nasdaq', '37460_TLX_nyse_mkt', '86915_RMIX_nasdaq', '77262_TTES_nasdaq', '88836_GNVC_nasdaq', '87649_CHRD_nasdaq', '83145_IRIX_nasdaq', '84562_HGRD_nasdaq', '87762_CRYP_nasdaq', '17778_BRK_nyse', '89929_FLCN_nasdaq', '87663_IPK_nyse_mkt', '89495_TAP_nyse', '76691_AETC_nasdaq', 'CASH']

    #settings['markets'] = settings['markets'][-20:]
    print(settings['markets'])
    
    assert settings['markets'][-1] == 'CASH', "Cash is last position in markets for causality calculation!"
    return settings



if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    # joblib.dump(results, 'saved_data/results.pkl')
    print(results['stats'])
