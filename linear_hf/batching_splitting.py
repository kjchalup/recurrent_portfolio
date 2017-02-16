import numpy as np
from preprocessing import non_nan_markets
from quantiacsToolbox import loadData

def draw_timeseries_batch(all_data, market_data, horizon, batch_size, batch_id, randseed=1):
    """ Make batches of data.

    Args:
        all_data: the data which the neural net uses to output a portfolio.
        market_data: the data the neural net uses to score a portfolio. open,close,high,low
        n_for_sharpe: the amount of portfolios output to use for gradient calculation.
        batch_size: the number of batches per epoch.
        batch_id: the batch id per batch.
        randseed: the epoch number, helps randomize between epochs.
    
    Returns:
        all_batch (n_batchsize, n_timesteps, data): batches for input data to neural net.
        market_batch (n_batchsize, n_timesteps, market_data): batches for scoring for neural net.
    """
    next_seed=  np.random.randint(0, 100000)
    np.random.seed(randseed)
    #import pdb;pdb.set_trace()
    perm_ids = np.random.permutation(all_data.shape[0]-horizon+1)
    np.random.seed(next_seed)
    #import pdb;pdb.set_trace()
    if (batch_id + 1) * batch_size > perm_ids.size:
        raise IndexError('Cant make this many batches, not enough data!')
    all_batch = np.zeros((batch_size, horizon, all_data.shape[1])).astype(np.float32)
    market_batch = np.zeros((batch_size, horizon, market_data.shape[1])).astype(np.float32)
    start_ids = perm_ids[batch_id * batch_size : (batch_id + 1) * batch_size]
    for point_id, start_id in enumerate(start_ids):
        all_batch[point_id, :, :] = all_data[start_id: start_id+horizon]
        market_batch[point_id, :, :] = market_data[start_id: start_id+horizon]
    return all_batch, market_batch

def split_validation_training(all_data, market_data, valid_period, horizon, n_for_sharpe, batch_id, batch_size, randseed):
    """ Splits validation and training, returns new batches for every epoch.
    
    Args:
        all_data: the data which the neural net uses to output a portfolio.
        market_data: the data the neural net uses to score a portfolio. open,close,high,low
        valid_period: number of batches of validation data; taken from newest times in all_data.
        horizon: size of total horizon used to predict n_for_sharpe
        n_for_sharpe: the amount of portfolios output to use for gradient calculation.
        batch_size: the number of batches per epoch.
        batch_id: the batch id per batch. should be for batch_id in range(batches_per_epoch)
        randseed: the epoch number, randomize between epochs. should be for epoch_id in range(num_epochs)

        This argument is called for every batch_id.
        batches_per_epoch calculated as follows: int(np.floor((all_data.shape[0]-horizon-2*n_ofr_sharpe-valid_period+1)/float(batch_size)))
        For validation data, the batch_id is set to 0, and the randseed is set to 1, so it will always return the same validation data.
        For time indexing, all_data is indexed -1 from market_data so that the positions are predicted from all_data, and scored against market_data.
    """
    
    all_val = None
    market_val = None
    if valid_period > 0:
        #import pdb;pdb.set_trace()
        all_val, market_val = draw_timeseries_batch(all_data=all_data[-valid_period-horizon-n_for_sharpe+1:-1], 
            market_data=market_data[-valid_period-horizon-n_for_sharpe+2:], 
            horizon=horizon+n_for_sharpe-1, 
            batch_size=valid_period,
            batch_id=0, randseed=1)
        market_val = market_val[:, -n_for_sharpe:, :]
        # ASSUME THAT valid_period is a divisor of batch_size!
        #import pdb;pdb.set_trace()
        if batch_size % valid_period != 0:
            raise ValueError, 'valid_period must be a divisor of batch_size!'
        all_val = np.tile(all_val, [batch_size/valid_period, 1, 1])
        market_val = np.tile(market_val, [batch_size/valid_period, 1, 1])
        
    all_batch, market_batch = draw_timeseries_batch(all_data=all_data[:-valid_period-n_for_sharpe-1] if valid_period > 0 else all_data[:-1], 
        market_data=market_data[1:-valid_period-n_for_sharpe] if valid_period > 0 else market_data[1:], 
        horizon=horizon+n_for_sharpe-1, 
        batch_size=batch_size, 
        batch_id=batch_id, randseed=randseed)
    market_batch = market_batch[:, -n_for_sharpe:, :]
    
    return all_val, market_val, all_batch, market_batch
