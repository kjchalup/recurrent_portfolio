""" Neural network (really just a linear network) routines. """
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as tf_rnn
from tensorflow import nn as tf_nn

from rnn_portfolio.costs import sharpe_tf

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
    lstm_cell = tf_rnn.BasicLSTMCell(num_units=n_markets, state_is_tuple=True)
    cell_state = tf.placeholder(TF_DTYPE, [None, lstm_cell.state_size[0]])
    hidden_state = tf.placeholder(TF_DTYPE, [None, lstm_cell.state_size[0]])
    init_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
    out, state_out_tf = tf_nn.dynamic_rnn(
        cell=lstm_cell, inputs=batch_in_tf, time_major=False, 
        sequence_length=seq_lens_tf, initial_state=init_state, dtype=tf.float32)
    
    if allow_shorting:
        out = out / tf.reduce_sum(tf.abs(out), axis=2, keep_dims=True)
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

    def __init__(self, n_ftrs, n_markets, n_time, n_sharpe, lbd=0.001, 
                 seed=None, allow_shorting=True):
        """ Initialize the regressor.

        Args:
          n_ftrs (float): Number of input features.
          n_markets (float): Number of markets (== number of outputs/4).
          n_time (float): Timesteps in batches.
          n_sharpe (float): Use this many timesteps to predict each
            position vector.
          lbd (float): l1 penalty coefficient.
          seed (int): Graph-level random seed, for testing purposes.
          allow_shorting (bool): If True, allow negative positions.
        """
        self.n_ftrs = n_ftrs
        self.n_markets = n_markets
        self.n_time = n_time
        self.n_sharpe = n_sharpe
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
            allow_shorting=allow_shorting)
        self.positions_tf = rnn_defs[0]
        self.state_out_tf = rnn_defs[1]
        self.state_in_tf = rnn_defs[2]
        self.last_state = [np.zeros((1, self.n_markets), dtype=NP_DTYPE),
                           np.zeros((1, self.n_markets), dtype=NP_DTYPE)]

        # Define the unnormalized loss function.
        self.loss_tf = -sharpe_tf(
            self.positions_tf[:, -self.n_sharpe:, :], self.batch_out_tf)

        # Define the optimizer.
        # raw_grads = tf.gradients(self.loss_tf, tf.trainable_variables())
        # grads = [tf.clip_by_value(g, -1, 1) for g in raw_grads]
        # self.train_op_tf = tf.train.AdamOptimizer(
        #     self.lr_tf).apply_gradients(zip(grads, tf.trainable_variables()))
        self.train_op_tf = tf.train.AdamOptimizer(
            learning_rate=self.lr_tf).minimize(self.loss_tf)

        # Define the saver that will serialize the weights/biases.
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a Tf session and initialize the variables.
        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)

    def restart_variables(self):
        c_init = np.zeros((1, self.n_markets), dtype=NP_DTYPE)
        h_init = np.zeros((1, self.n_markets), dtype=NP_DTYPE)
        self.last_state = (c_init, h_init)
        self.sess.run(self.init_op)

    def get_weights(self):
        raise NotImplementedError

    def get_biases(self):
        raise NotImplementedError

    def _positions_np(self, batch_in):
        """ Predict a portfolio for a training batch.
        NOTE: this does not update the internal state.

        Args:
          data_in (n_batch, n_time, n_ftrs): Input data.

        Returns:
          positions (n_batch, n_markets): Positions.
        """
        seq_lens = [batch_in.shape[1]] * batch_in.shape[0]
        positions, state =  self.sess.run([self.positions_tf, 
                                           self.state_out_tf],
                                 {self.state_in_tf[0]: self.last_state[0],
                                  self.state_in_tf[1]: self.last_state[1],
                                  self.seq_lens_tf: seq_lens,
                                  self.batch_in_tf: batch_in})
        return positions


    def predict(self, data_in):
        """ Predict a portfolio for a test batch.
        NOTE: this updates the internal state.

        Args:
            data_in (n_time, n_ftrs): Input data, prices just for the last day. 
                Since the state is copied through, each test day processes just
                one time-point.

        Returns:
            positions (n_markets): Positions.
        """
        seq_lens = [1]
        data_in = np.tile(data_in, [self.n_time, 1])
        feed_dict = {self.state_in_tf[0]: self.last_state[0],
                     self.state_in_tf[1]: self.last_state[1],
                     self.seq_lens_tf: seq_lens,
                     self.batch_in_tf: data_in[None, :, :]}
        positions, state =  self.sess.run([self.positions_tf, 
                                           self.state_out_tf],
                                          feed_dict)
        self.last_state = state
        return positions[0, 0]

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
        c_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        h_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        seq_lens = batch_in.shape[0] * [batch_in.shape[1]]
        return self.sess.run(self.loss_tf,
                             {self.batch_in_tf: batch_in,
                              self.batch_out_tf: batch_out,
                              self.state_in_tf[0]: c_init,
                              self.state_in_tf[1]: h_init,
                              self.seq_lens_tf: seq_lens})
    
    def train_step(self, batch_in, batch_out, lr):
        """ Do one gradient-descent step. """
        c_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        h_init = np.zeros((batch_in.shape[0], self.n_markets), dtype=NP_DTYPE)
        seq_lens = batch_in.shape[0] * [batch_in.shape[1]]
        self.sess.run(self.train_op_tf,
                      {self.batch_in_tf: batch_in,
                       self.batch_out_tf: batch_out,
                       self.seq_lens_tf: seq_lens,
                       self.state_in_tf[0]: c_init,
                       self.state_in_tf[1]: h_init,
                       self.lr_tf: lr})

    def save(self, fname='saved_data/rnnmodel'):
        self.save_path = self.saver.save(self.sess, fname)

    def load(self):
        self.saver.restore(self.sess, self.save_path)
