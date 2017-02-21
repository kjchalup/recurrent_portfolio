""" Neural network (really just a linear network) routines. """
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow import nn as tf_nn

from linear_hf.costs import sharpe_tf
from linear_hf.costs import sharpe_onepos_tf
from linear_hf import strategies

from . import TF_DTYPE
from . import NP_DTYPE

def define_rnn(batch_in_tf, seq_lens_tf, n_sharpe,
              n_time, n_ftrs, n_markets, allow_shorting=True, equality=False):
    """ Define a neural net for the Linear regressor.

    Args:
      batch_in_tf (n_batch, n_time, n_ftrs): Input data.
      seq_lens_tf (n_batch): Lengths of each batch sequence. Pad with zeros
        afterwards.
      state_in_tf: Symbolic init state. Can be None or returned by the rnn.
      n_sharpe (float): How many position-outputs to compute.
      n_time (float): Number of timesteps for input data.
      n_ftrs (float): Number of input features.
      W (n_ftrs * (n_time-n_sharpe+1), n_markets): Weight matrix.
      b (n_markets): Biases.
      zero_thr (scalar): Set smaller weights to zero.

    Returns:
      positions (n_batch, n_sharpe, n_markets): Positions for each market.
    """
    if equality:
        assert allow_shorting, 'equality nets only possible with shorting!'
    lstm_cell = tf_rnn.BasicLSTMCell(num_units=n_markets, state_is_tuple=True)
    cell_state = tf.placeholder(TF_DTYPE, [None, lstm_cell.state_size[0]])
    hidden_state = tf.placeholder(TF_DTYPE, [None, lstm_cell.state_size[0]])
    init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
    out, state_out_tf = tf_nn.dynamic_rnn(cell=lstm_cell, inputs=batch_in_tf, time_major=False, 
        sequence_length=seq_lens_tf, initial_state=init_state, dtype=tf.float32)
    if allow_shorting:
        out = out / tf.reduce_sum(tf.abs(out), axis=2, keep_dims=True)
    elif equality:
        out_pos = tf.sigmoid(3*out)
        out_neg = tf.sigmoid(-3*out)
        out_pos_sum = tf.reduce_sum(out_pos, axis=2, keep_dims=True)
        out_neg_sum = tf.reduce_sum(-out_neg, axis=2, keep_dims=True)
        out = out_pos * out_neg_sum / out_pos_sum + out_neg
        out /= tf.reduce_sum(tf.abs(out), axis=2, keep_dims=True)
    else:
        out = tf.pow(out, 2)
        out = out / tf.reduce_sum(out, axis=2, keep_dims=True)
    return out, state_out_tf, (cell_state, hidden_state)

class RNN(object):
    """ A linear, L1-regularized position predictor.

    This predictor will scan the input batch using a shared
    set of linear weights. It will then output a vector of
    positions whose absolute values sum to one.
    """

    def __init__(self, n_ftrs, n_markets, n_time, 
                 n_sharpe, W_init=None, lbd=0.001,
                 causality_matrix=None, n_csl_ftrs=None, seed=None,
                 allow_shorting=True, cost='sharpe'):
        """ Initialize the regressor.

        Args:
          n_ftrs (float): Number of input features.
          n_markets (float): Number of markets (== number of outputs/4).
          n_time (float): Timesteps in batches.
          n_sharpe (float): Use this many timesteps to predict each
            position vector.
          W_init (n_ftrs * (n_time-n_sharpe+1), n_markets): Weight
            initalization.
          lbd (float): l1 penalty coefficient.
          causality_matrix (n_ftrs, n_markets): A matrix where the [ij]
            entry is positive if market corresponding to feature i seems
            to cause changes in market j. Used to decrease the L1 penalty
            on causally meaningful weights.
          seed (int): Graph-level random seed, for testing purposes.
          allow_shorting (bool): If True, allow negative positions.
          cost (str): cost to use: 'sharpe', 'min_return', 'mean_return', or 'mixed_return'
        """
        self.n_ftrs = n_ftrs
        self.n_markets = n_markets
        self.n_time = n_time
        self.n_sharpe = n_sharpe
        self.horizon = n_time - n_sharpe + 1
        self.lbd = lbd
        self.rnn_state = None

        # Doefine symbolic placeholders for data batches.
        self.batch_in_tf = tf.placeholder(
            TF_DTYPE, shape=[None, n_time, n_ftrs],
            name='input_batch')
        self.batch_out_tf = tf.placeholder(
            TF_DTYPE, shape=[None, n_sharpe, n_markets * 4],
            name='output_batch')
        self.seq_lens_tf = tf.placeholder(
            tf.int32, shape=[None], name='sequence_lengths')

        # Neural net training-related placeholders.
        self.lr_tf = tf.placeholder(
            TF_DTYPE, name='learning_rate')

        # Define the position outputs on a batch of timeseries.
        rnn_defs = define_rnn(
            self.batch_in_tf, self.seq_lens_tf, n_sharpe=n_sharpe, 
            n_time=n_time, n_ftrs=n_ftrs, n_markets=self.n_markets,
            allow_shorting=allow_shorting,
            equality=cost.startswith('equality'))
        self.positions_tf = rnn_defs[0]
        self.state_out_tf = rnn_defs[1]
        self.state_in_tf = rnn_defs[2]
        self.last_state = [np.zeros((1, self.n_markets), dtype=NP_DTYPE),
                           np.zeros((1, self.n_markets), dtype=NP_DTYPE)]

        # Define the L1 penalty, taking causality into account.
        self.l1_penalty_tf = tf.reduce_sum(tf.zeros([2,2]))

        # Define the unnormalized loss function.
        if cost.startswith('onepos'):
            self.loss_tf = -sharpe_onepos_tf(
                self.positions_tf, self.batch_out_tf, n_sharpe,
                n_markets, cost=cost) + self.l1_penalty_tf
        else:
            self.loss_tf = -sharpe_tf(
                self.positions_tf, self.batch_out_tf,
                n_sharpe, n_markets, cost=cost) + self.l1_penalty_tf

        # Define the optimizer.
        self.train_op_tf = tf.train.AdamOptimizer(
            learning_rate=self.lr_tf).minimize(self.loss_tf)

        # Define the saver that will serialize the weights/biases.
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a Tf session and initialize the variables.
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def restart_variables(self):
        self.sess.run(self.init_op)

    def get_weights(self):
        raise NotImplementedError

    def get_biases(self):
        raise NotImplementedError

    def _positions_np(self, batch_in):
        """ Predict a portfolio for a training batch.

        Args:
          batch_in (n_batch, n_time, n_ftrs): Input data.

        Returns:
          positions (n_batch, n_markets): Positions.
        """
        seq_lens = batch_in.shape[0] * [batch_in.shape[1]]
        return self.sess.run(self.positions_tf,
                             {self.batch_in_tf: batch_in, 
                              self.seq_lens_tf: seq_lens})

    def predict(self, data_in):
        """ Predict a portfolio for a test batch.

        Args:
          data_in (horizon, n_ftrs): Input data, where
            horizon = n_time - n_sharpe + 1. This corresponds
            to data needed to predict just one portfolio.

        Returns:
          positions (n_markets): Positions.
        """
        if np.sum(self.last_state[0]) == 0:
            # The rnn is freshly retrained. Run it through the whole batch.   
            seq_lens = [data_in.shape[1]]
            positions, state =  self.sess.run([self.positions_tf, self.state_out_tf],
                                     {self.state_in_tf[0]: self.last_state[0],
                                      self.state_in_tf[1]: self.last_state[1],
                                      self.seq_lens_tf: seq_lens,
                                      self.batch_in_tf: 
                                      np.expand_dims(data_in, 0)})
            self.last_state = state
            return positions[0, -1]
        else:
            # We're inside a prediction interval. Only run the rnn on the last timestep.
            seq_lens = [1]
            positions, state =  self.sess.run([self.positions_tf, self.state_out_tf],
                                     {self.state_in_tf[0]: self.last_state[0],
                                      self.state_in_tf[1]: self.last_state[1],
                                      self.seq_lens_tf: seq_lens,
                                      self.batch_in_tf: np.expand_dims(data_in, 0)})
            self.last_state = state
            return positions[0, 0]

    def l1_penalty_np(self):
        """ Compute the L1 penalty on the weights. """
        return self.sess.run(self.l1_penalty_tf)

    def loss_np(self, batch_in, batch_out):
        """ Compute the current Sharpe loss.

        Args:
          batch_in (n_batch, n_time, n_ftrs): Input data.
          batch_out (n_batch, n_sharpe, n_markets * 4): Open,
            close, high and low prices.

        Returns:
          loss (float): the average negative Sharpe ratio of the
            current strategy.
        """
        seq_lens = batch_in.shape[0] * [batch_in.shape[1]]
        c_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        h_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        return self.sess.run(self.loss_tf,
                             {self.batch_in_tf: batch_in,
                              self.batch_out_tf: batch_out,
                              self.state_in_tf[0]: c_init,
                              self.state_in_tf[1]: h_init,
                              self.seq_lens_tf: seq_lens})
    
    def train_step(self, batch_in, batch_out, lr):
        """ Do one gradient-descent step. """
        seq_lens = batch_in.shape[0] * [batch_in.shape[1]]
        c_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        h_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        self.sess.run(self.train_op_tf,
                      {self.batch_in_tf: batch_in,
                       self.batch_out_tf: batch_out,
                       self.seq_lens_tf: seq_lens,
                       self.state_in_tf[0]: c_init,
                       self.state_in_tf[1]: h_init,
                       self.lr_tf: lr})
        # Reset the internal rnn state!
        self.last_state = [np.zeros((1, self.n_markets), dtype=NP_DTYPE),
                           np.zeros((1, self.n_markets), dtype=NP_DTYPE)]

    def save(self, fname='saved_data/rnnmodel'):
        self.save_path = self.saver.save(self.sess, fname)

    def load(self):
        self.saver.restore(self.sess, self.save_path)
