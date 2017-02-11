import os
import sys
import joblib

import numpy as np
import tensorflow as tf

from costs import sharpe_tf

def define_nn(batch_in_tf, n_sharpe, 
              n_time, n_ftrs, W, b, allow_shorting):
    """ Define a neural net for the Linear regressor.
    
    Args:
      batch_in_tf (n_batch, n_time, n_ftrs): Input data.
      n_sharpe (float): How many position-outputs to compute.
      n_time (float): Number of timesteps for input data.
      n_ftrs (float): Number of input features.
      W (n_ftrs * (n_time-n_sharpe+1), n_markets): Weight matrix.
      b (n_markets): Biases.
      zero_thr (scalar): Set smaller weights to zero.

    Returns:
      positions (n_batch, n_sharpe, n_markets): Positions for each market
    """
    horizon = n_time - n_sharpe + 1
    def apply_net(x):
        out = tf.add(tf.matmul(x, W), b)
        if allow_shorting:
            out = out / tf.reduce_sum(tf.abs(out), axis=1, keep_dims=True)
        else:
            out = tf.pow(out, 2)
            out = out / tf.reduce_sum(out, axis=1, keep_dims=True)
        return out

    positions = []
    for t_id in range(n_sharpe):
        positions.append(apply_net(tf.reshape(
            batch_in_tf[:, t_id:t_id+horizon, :], 
            (-1, n_ftrs * horizon))))
    
    return tf.transpose(positions, [1, 0, 2])

class Linear(object):
    """ A linear, L1-regularized position predictor.
    
    This predictor will scan the input batch using a shared
    set of linear weights. It will then output a vector of
    positions whose absolute values sum to one.
    """
    
    def __init__(self, n_ftrs, n_markets, n_time, 
                 n_sharpe, W_init=None, lbd=0.001, 
                 causality_matrix=None, n_csl_ftrs=None, seed=None,
                 allow_shorting=True):
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
        """
        self.n_ftrs = n_ftrs
        self.n_markets = n_markets
        self.n_time = n_time
        self.n_sharpe = n_sharpe
        self.horizon = n_time - n_sharpe + 1
        self.lbd = lbd
        
        # Doefine symbolic placeholders for data batches.
        self.batch_in_tf = tf.placeholder(
            tf.float32, shape=[None, n_time, n_ftrs], 
            name='input_batch')
        self.batch_out_tf = tf.placeholder(
            tf.float32, shape=[None, n_sharpe, n_markets * 4],
            name='output_batch')
        self.test_in_tf = tf.placeholder(
            tf.float32, shape=[self.horizon, n_ftrs], 
            name='single_datapoint')

        # Neural net training-related placeholders.
        self.lr_tf = tf.placeholder(
            tf.float32, name='learning_rate')

        # Define nn weights and biases.
        if W_init is None:
            W_init = tf.truncated_normal(
                [n_ftrs * self.horizon, n_markets], 
                stddev=1. / (n_ftrs * self.horizon))
        self.W = tf.Variable(W_init, name='nn_weights')
        self.b = tf.Variable(tf.zeros(n_markets), 
                             name='nn_biases')

        # Causality matrix: small for the causal entries. Clone it across 
        # the timesteps (so each market is either causal or not, at all timesteps).

        # Define the position outputs on a batch of timeseries.
        self.positions_tf = define_nn(self.batch_in_tf, 
                                      n_sharpe=n_sharpe, 
                                      n_time=n_time,
                                      n_ftrs=n_ftrs, 
                                      W=self.W, b=self.b,
                                      allow_shorting=allow_shorting)

        # Define the position output on one stock timeseries.
        prediction_tf = tf.add(tf.matmul(tf.reshape(
            self.test_in_tf, (1, n_ftrs * self.horizon)), 
                                         self.W), self.b)[0]
        if allow_shorting:
            self.prediction_tf = prediction_tf / tf.reduce_sum(
                tf.abs(prediction_tf))
        else:
            prediction_tf = tf.pow(prediction_tf, 2)
            self.prediction_tf = prediction_tf / tf.reduce_sum(
                prediction_tf)

        # Define the L1 penalty, taking causality into account.
        if causality_matrix is None:
            self.l1_penalty_tf = self.lbd * tf.reduce_sum(tf.abs(self.W))
        else:
            self.causality_matrix = np.tile(causality_matrix, [self.horizon, 1])
            # self.l1_penalty_tf = self.lbd * tf.reduce_sum(tf.abs(
            #     self.W * (1-self.causality_matrix)))
            self.l1_penalty_tf = self.lbd * tf.reduce_sum(tf.abs(
                tf.boolean_mask(self.W, self.causality_matrix==0)))

        # Define the unnormalized loss function.
        self.loss_tf = -sharpe_tf(self.positions_tf, self.batch_out_tf, 
                                 n_sharpe, n_markets) + self.l1_penalty_tf
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
        return self.sess.run(self.W)

    def _positions_np(self, batch_in):
        """ Predict a portfolio for a training batch.

        Args:
          batch_in (n_batch, n_time, n_ftrs): Input data.
        
        Returns:
          positions (n_batch, n_markets): Positions.
        """
        return self.sess.run(self.positions_tf, {self.batch_in_tf:
                                            batch_in})

    def predict(self, data_in):
        """ Predict a portfolio for a test batch.
        
        Args:
          data_in (horizon, n_ftrs): Input data, where
            horizon = n_time - n_sharpe + 1. This corresponds
            to data needed to predict just one portfolio.
        
        Returns:
          positions (n_markets): Positions.
        """
        return self.sess.run(self.prediction_tf, 
                             {self.test_in_tf: data_in})

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
        return self.sess.run(self.loss_tf, 
                             {self.batch_in_tf: batch_in, 
                              self.batch_out_tf: batch_out})
        
    def train_step(self, batch_in, batch_out, lr):
        self.sess.run(self.train_op_tf, 
                      {self.batch_in_tf: batch_in,
                       self.batch_out_tf: batch_out,
                       self.lr_tf: lr})

    def save(self, fname='saved_data/model'):
        self.save_path = self.saver.save(self.sess, fname)

    def load(self):
        self.saver.restore(self.sess, self.save_path)
    


