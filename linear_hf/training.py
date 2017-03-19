""" Routines for training. """
import sys
import numpy as np
from linear_hf import rnn
from linear_hf import neuralnet
from linear_hf.preprocessing import split_val_tr
from linear_hf.preprocessing import get_n_batch

def lr_calc(settings, epoch_id):
    # Update the learning rate, exponential schedule
    lr_mult = settings['lr_mult_base'] ** (1. / settings['num_epochs'])
    lr_new = settings['lr'] * lr_mult ** epoch_id
    return lr_new

def loss_calc(settings, all_batch, market_batch):
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

def update_nn(settings, best_sharpe, epoch_sharpe):
    """ Saves neural net and updates best_sharpe if better in this epoch.

    Args:
        settings: contains the neuralnet
        best_sharpe: the previously highest sharpe (or cost function)
        epoch_sharpe: the sharpe for the current epoch (either validation or avg)

    Returns:
        settings: saved neuralnet in settings
        best_sharpe: updated new sharpe or old best sharpe
    """
    if epoch_sharpe > best_sharpe:
        best_sharpe = epoch_sharpe
        settings['nn'].save()

    return settings, best_sharpe


def init_nn(settings, n_ftrs):
    """ Intializes the neural net

    Args:
        settings: where the neuralnet gets initialized
        n_ftrs: size of the neuralnet input layer
        nn_type: type of neural net
    Returns:
        settings: a dict with ['nn'] which is the initialized neuralnet.
    """
    if settings['nn_type'] == 'linear':
        settings['nn'] = neuralnet.Linear(n_ftrs=n_ftrs,
                                          n_markets=len(settings['markets']),
                                          n_time=settings['n_time'],
                                          n_sharpe=settings['n_sharpe'],
                                          lbd=settings['lbd'],
                                          allow_shorting=settings['allow_shorting'])
    elif settings['nn_type'] == 'rnn':
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
    batches_per_epoch = get_n_batch(all_data.shape[0], settings['horizon'],
                                    settings['val_period'], settings['n_sharpe'],
                                    settings['batch_size'])

    for epoch_id in range(settings['num_epochs']):
        seed = np.random.randint(10000)
        tr_sharpe = 0.
        tr_scores = []
        val_sharpe = 0.
        lr_new = lr_calc(settings, epoch_id)
        # Train an epoch.
        for batch_id in range(batches_per_epoch):
            # Split data into validation and training batches.
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
            settings, best_val_sharpe = update_nn(
                settings, best_val_sharpe, val_sharpe)
        else:
            settings, best_tr_sharpe = update_nn(
                settings, best_tr_sharpe, tr_sharpe)

        # Write out data for epoch.
        sys.stdout.write('\nEpoch {}, val/tr Sharpe {:.4}/{:.4g}.'.format(
            epoch_id, val_sharpe, min(tr_scores)))
        sys.stdout.flush()

    return settings

