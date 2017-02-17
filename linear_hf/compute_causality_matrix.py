import numpy as np
import joblib

from linear_hf.causality import causal_matrix
from linear_hf.preprocessing import preprocess


def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL,CLOSE_LASTTRADE, 
    CLOSE_ASK, CLOSE_BID, RETURN, SHARE, DIVIDEND, TOTALCAP, exposure, equity, settings, fundEquity):
    market_data, all_data, should_retrain = preprocess(
        settings['markets'], OPEN, CLOSE, HIGH, LOW, VOL, DATE,
        CLOSE_LASTTRADE, CLOSE_ASK, CLOSE_BID, RETURN, SHARE,
        DIVIDEND, TOTALCAP, postipo=100, filler=0.123456789)
    opens = market_data[:, :len(settings['markets'])-1]
    cm = causal_matrix(opens, verbose=False, method='nearest',
                       n_neighbors=30, nruns=100, max_data=1000)
    # Add cash
    cm_withcash = np.zeros((cm.shape[0]+1, cm.shape[1]+1))
    cm_withcash[:cm.shape[0], :cm.shape[1]] = cm
    cm_withcash[-1, -1] = 1.

    joblib.dump([settings, cm_withcash], 'saved_data/causality_matrix.pkl')
    return None

def mySettings():
    settings={}
    # Futures Contracts
    settings['lookback'] = 1000
    settings['budget'] = 10**6
    settings['slippage'] = 0.05
    settings['beginInSample'] = '20040101'
    settings['endInSample'] = '20140101'

    # Only keep markets that have not died out by beginInSample.
    settings['markets'] = ['85653_AI_nyse', '81061_MCK_nyse', '62148_CSX_nyse', '85621_MTD_nyse', '85442_TSM_nyse', '36468_SHW_nyse', '50788_ESL_nyse', '46690_CBT_nyse', '86023_MUS_nyse', '77925_MQT_nyse', '77823_KMP_nyse', '32678_HEI_nyse', '87198_ARLP_nasdaq', '87289_SNH_nyse', '77971_EFII_nasdaq', '77860_JEQ_nyse', '55001_TRN_nyse', '79323_ALL_nyse', '47941_TGNA_nyse', '46463_GFF_nyse', '51633_VVC_nyse', '80691_LPT_nyse', '75429_PHF_nyse_mkt', '77604_ALU_nyse', '22840_HSH_nyse', '75183_PCF_nyse', '19350_DE_nyse', '78034_PDCO_nasdaq', '38762_NI_nyse', '75278_AB_nyse', '49656_BK_nyse', '11369_UBSI_nasdaq', '79665_TEI_nyse', '24328_EQT_nyse', '76697_HNT_nyse', '61778_KYO_nyse', '82924_JW_nyse', '84042_PAG_nyse', '83604_SKM_nyse', '55213_RT_nyse', '86143_VVR_nyse', '70826_MFM_nyse', '85421_CHL_nyse', '59504_BTI_nyse_mkt', '14816_TR_nyse', '79363_AZN_nyse', '68187_WRI_nyse', '73139_SYK_nyse', '86102_FII_nyse', '77078_TOT_nyse', '78775_CREAF_nasdaq', '88249_SCRX_nasdaq', '85074_LQ_nyse', '79195_PERY_nasdaq', '84351_BHBC_nasdaq', '79795_ASCA_nasdaq', '78758_BBHL_nasdaq', '86290_CVV_nasdaq', '89556_MPQ_nyse_mkt', '12236_NOVN_nasdaq', '77420_FRED_nasdaq', '89374_PCO_nyse', '80122_FFLC_nasdaq', '11634_RBNC_nasdaq', '87268_CIR_nyse', '81705_FEIC_nasdaq', '76515_DHC_nyse_mkt', '11292_ICCC_nasdaq', '88784_ADLR_nasdaq', '23799_CIA_nyse', '91265_HAXS_nasdaq', '69892_GG_nyse', '89103_GLAD_nasdaq', '82261_GSIG_nasdaq', '85686_DEPO_nasdaq', '76037_CBC_nyse', '89928_FBTX_nasdaq', '88550_OPNT_nasdaq', '38172_WOC_nyse_mkt', '77555_REM_nyse', '86313_UBA_nyse', '80539_NKTR_nasdaq', '86839_OBAS_nasdaq', '83533_MXIC_nasdaq', '82207_TRKN_nasdaq', '79358_VMV_nyse_mkt', '85510_TONS_nasdaq', '37460_TLX_nyse_mkt', '86915_RMIX_nasdaq', '77262_TTES_nasdaq', '88836_GNVC_nasdaq', '87649_CHRD_nasdaq', '83145_IRIX_nasdaq', '84562_HGRD_nasdaq', '87762_CRYP_nasdaq', '17778_BRK_nyse', '89929_FLCN_nasdaq', '87663_IPK_nyse_mkt', '89495_TAP_nyse', '76691_AETC_nasdaq', 'CASH']
    return settings

if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    print results['stats']
