""" Routines for training the rnn. """
import sys
import numpy as np
from linear_hf import rnn
from linear_hf.preprocessing import split_val_tr
from linear_hf.preprocessing import get_n_batch

def make_empty_datadict(markets, max_time=10000):
    """ Make a dictionary containing placeholders for OPEN, CLOSE,
    HIGH, LOW and DATE data that will be filled up as training continues.
    """
    data = {}
    for key in ['OPEN', 'CLOSE', 'HIGH', 'LOW']:
        data[key] = -np.pi * np.ones((max_time, len(markets)))
    data['DATE'] = -np.pi * np.ones((max_time))
    return data


def append_new_data(past_data, OPEN, CLOSE, HIGH, LOW, DATE):
    """ Append the OPEN, ..., DATE data to their respective arrays
    in the past_data dictionary, assuming this is done on each timestep.

    Args:
        past_data (dict): Dictionary containing 'OPEN', ..., 'DATE' as
            keys and nan-padded value arrays as values.
        OPEN, CLOSE, HIGH, LOW, DATE: Quantiacs data arrays.

    Returns:
        OPEN, ..., DATE: Data with pre-pended past entries from the dict.
            The dictionary is also updated with the new data.
    """
    keys = ['OPEN', 'CLOSE', 'HIGH', 'LOW', 'DATE']
    first_empty = np.where(past_data['OPEN'] == -np.pi)[0][0]
    lookback = OPEN.shape[0]
    for key in keys:
        if first_empty == 0:
            # This is the first iteration!
            past_data[key][:lookback] = locals()[key]
        else:
            past_data[key][
                first_empty - lookback + 1 : first_empty + 1] = locals()[key]
    if first_empty == 0:
        return [past_data[key][:lookback] for key in keys]
    else:
        return [past_data[key][: first_empty + 1] for key in keys]


def lr_calc(settings, epoch_id):
    """Update the learning rate with an exponential schedule."""
    lr_mult = settings['lr_mult_base'] ** (1. / settings['num_epochs'])
    lr_new = settings['lr'] * lr_mult ** epoch_id
    return lr_new


def loss_calc(settings, all_batch, market_batch):
    """ Calculates nn's NEGATIVE loss.

    Args:
        settings: contains the neural net
        all_batch: the inputs to neural net
        market_batch: [open close high low] used to calculate loss
    Returns:
        cost: loss - l1 penalty
    """
    loss = settings['nn'].loss_np(all_batch, market_batch)
    return -loss


def init_nn(settings, n_ftrs):
    """ Intializes the neural net

    Args:
        settings: where the neuralnet gets initialized
        n_ftrs: size of the neuralnet input layer
    Returns:
        settings: a dict with ['nn'] which is the initialized neuralnet.
    """
    settings['nn'] = rnn.RNN(n_ftrs=n_ftrs,
                             n_markets=len(settings['markets']),
                             n_time=settings['n_time'],
                             n_sharpe=settings['n_sharpe'],
                             allow_shorting=settings['allow_shorting'])
    return settings

def train(settings, all_data, market_data):
    """ Trains the neuralnet.
    Total steps:
    1) train for settings['num_epochs']
        a) calculates new learning rate for each epoch
    2) saves neural net if the epoch has a better val_sharpe or tr_sharpe
    3) saves the best_val_sharpe to settings['best_val_sharpe']

    Args:
        settings: contains nn to be trained, as well as other settings
        all_data: total data fed into neuralnet (ntimesteps, nftrs)
        market_data: data to score neuralnet (ntimesteps, nmarkets*4)
    Returns:
        settings: updated neural net, and best_val_sharpe
    """
    best_val_sharpe = -np.inf
    best_tr_sharpe = -np.inf
    batches_per_epoch = get_n_batch(all_data.shape[0],
                                    settings['horizon'],
                                    settings['val_period'],
                                    settings['n_sharpe'],
                                    settings['batch_size'])

    for epoch_id in range(settings['num_epochs']):
        seed = np.random.randint(10000)
        tr_sharpe = 0.
        tr_scores = []
        val_sharpe = 0.
        lr_new = lr_calc(settings, epoch_id)
        # Train an epoch.
        for batch_id in range(batches_per_epoch):
            all_val, market_val, all_batch, market_batch = split_val_tr(
                all_data=all_data, market_data=market_data,
                valid_period=settings['val_period'],
                horizon=settings['horizon'],
                n_for_sharpe=settings['n_sharpe'],
                batch_id=batch_id,
                batch_size=settings['batch_size'],
                randseed=seed)

            # Train.
            settings['nn'].train_step(batch_in=all_batch,
                                      batch_out=market_batch, lr=lr_new)
            tr_score = loss_calc(settings, all_batch, market_batch)
            tr_sharpe += tr_score
            tr_scores.append(tr_score)

        # Calculate sharpes for the epoch
        tr_sharpe /= batches_per_epoch
        if settings['val_period'] > 0:
            val_sharpe = loss_calc(settings, all_batch=all_val,
                                   market_batch=market_val)

        # Update neural net, and attendant values if NN is better than previous.
        if settings['val_period'] > 0:
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                settings['nn'].save()
        else:
            if tr_sharpe > best_tr_sharpe:
                best_tr_sharpe = tr_sharpe
                settings['nn'].save()

        # Write out data for epoch.
        sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
            epoch_id, val_sharpe, min(tr_scores)))
        sys.stdout.flush()

    return settings
