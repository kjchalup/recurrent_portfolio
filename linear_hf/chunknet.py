""" Neural network (really just a linear network) routines. """
import numpy as np
import tensorflow as tf

from costs import sharpe_tf

def initialize_blockdiagonal(n_ftrs, n_time,
                             n_sharpe, n_markets, n_blocks):
    """ Initialize a list of diagonal blocks of a weight matrix.

    Args:
      n_ftrs, n_time, n_sharpe, n_markets: Corresponds to the parameters
        of the ChunkLinear object.
      n_blocks (int): Number of blocks to split the full weight matrix
        into. n_blocks MUST DIVIDE n_markets, and n_ftrs!

    Returns:
      blocks (list of tensors of shape
        (n_ftrs * (n_time-n_sharpe+1) / n_blocks , n_markets / n_blocks)):
        List of blocks of the diagonal weight matrix.
    """
    if n_markets % n_blocks != 0:
        raise ValueError('n_blocks must divide n_markets!')
    if n_ftrs % n_blocks != 0:
        raise ValueError('n_blocks must divide n_ftrs!')
    horizon = n_time - n_sharpe + 1
    blocks = [tf.truncated_normal((n_ftrs * horizon / n_blocks,
                                   n_markets / n_blocks),
                                  stddev=(float(n_blocks) /
                                          (n_ftrs * horizon)),
                                  dtype=tf.float32)
              for _ in range(n_blocks)]
    return blocks

def define_chunknn(batch_in_tf, n_sharpe,
                   n_time, n_ftrs, Ws, b, allow_shorting, shuffle):
    """ Define a neural net for the Linear regressor.

    Args:
      batch_in_tf (n_batch, n_time, n_ftrs): Input data.
      n_sharpe (float): How many position-outputs to compute.
      n_time (float): Number of timesteps for input data.
      n_ftrs (float): Number of input features.
      Ws (list of tensors of shape
        (n_ftrs * (n_time-n_sharpe+1) / n_blocks , n_markets / n_blocks)):
        List of blocks of the diagonal weight matrix.
      b (n_markets): Biases.
      allow_shorting (bool): If False, makes all positions positive.
      shuffle: A Python function used to shuffle each sub-batch of
        shape (n_batch, horizon, n_ftrs).

    Returns:
      positions (n_batch, n_sharpe, n_markets): Positions for each market
    """
    horizon = n_time - n_sharpe + 1
    def apply_net(x):
        """ Feed-forward x through the net. """
        outs = []
        for block_id, W in enumerate(Ws):
            block_size = tf.shape(W)[0]
            outs.append(tf.matmul(x[:, block_id * block_size:
                                    (block_id+1) * block_size], W))
        out = tf.add(tf.concat(outs, 1), b)
        if allow_shorting:
            out = out / tf.reduce_sum(tf.abs(out), axis=1, keep_dims=True)
        else:
            out = tf.pow(out, 2)
            out = out / tf.reduce_sum(out, axis=1, keep_dims=True)
        return out

    positions = []
    for t_id in range(n_sharpe):
        positions.append(apply_net(tf.reshape(
            shuffle(batch_in_tf[:, t_id:t_id+horizon, :]),
            (-1, n_ftrs * horizon))))

    return tf.transpose(positions, [1, 0, 2])

class ChunkLinear(object):
    """ A linear, L1-regularized position predictor with a block-dia-
    gonal weight matrix.

    This predictor will scan the input batch using a shared
    set of linear weights. It will then output a vector of
    positions whose absolute values sum to one.
    """

    def __init__(self, n_ftrs, n_markets, n_time,
                 n_sharpe, n_chunks, W_init=None, lbd=0.001,
                 causality_matrix=None,
                 allow_shorting=True, cost='sharpe'):
        """ Initialize the regressor.

        Args:
          n_ftrs (float): Number of input features.
          n_markets (float): Number of markets (== number of outputs/4).
          n_time (float): Timesteps in batches.
          n_sharpe (float): Use this many timesteps to predict each
            position vector.
          n_chunks (float): Size of non-zero blocks of the weight matrix.
          W_init: Initial value for the weight matrices. See
            initialize_blockdiagonal for its shape/meaning.
          lbd (float): l1 penalty coefficient.
          causality_matrix (n_ftrs, n_markets): A matrix where the [ij]
            entry is positive if market corresponding to feature i seems
            to cause changes in market j. Used to decrease the L1 penalty
            on causally meaningful weights.
          allow_shorting (bool): If True, allow negative positions.
          cost (str): cost to use: 'sharpe', 'min_return',
            'mean_return', or 'mixed_return'
        """
        self.n_ftrs = n_ftrs
        self.n_markets = n_markets
        self.n_time = n_time
        self.n_sharpe = n_sharpe
        self.n_chunks = n_chunks
        self.horizon = n_time - n_sharpe + 1
        self.lbd = lbd

        # Define symbolic placeholders for data batches.
        self.batch_in_tf = tf.placeholder(
            tf.float32, shape=[None, n_time, n_ftrs],
            name='input_batch')
        self.batch_out_tf = tf.placeholder(
            tf.float32, shape=[None, n_sharpe, n_markets * 4],
            name='output_batch')

        # Neural net training-related placeholders.
        self.lr_tf = tf.placeholder(
            tf.float32, name='learning_rate')

        # Define nn weights and biases.
        self.data_permutation = np.random.permutation(
            (self.horizon - 1) * self.n_ftrs).astype(np.int32)

        if W_init is None:
            W_init = initialize_blockdiagonal(
                self.n_ftrs, self.n_time, self.n_sharpe,
                self.n_markets, self.n_chunks)
        self.Ws = [tf.Variable(W_block, name='weights_block{}'.format(
            block_id)) for block_id, W_block in enumerate(W_init)]
        self.b = tf.Variable(tf.zeros(n_markets),
                             name='nn_biases')

        # Define the position outputs on a batch of timeseries.
        self.positions_tf = define_chunknn(self.batch_in_tf,
                                           n_sharpe=n_sharpe,
                                           n_time=n_time,
                                           n_ftrs=n_ftrs,
                                           Ws=self.Ws, b=self.b,
                                           allow_shorting=allow_shorting,
                                           shuffle=self.shuffle)

        # Define the L1 penalty, taking causality into account.
        if causality_matrix is None:
            self.l1_penalty_tf = self.lbd * tf.reduce_sum(tf.abs(self.Ws))
        else:
            self.causality_matrix = np.tile(causality_matrix, [self.horizon, 1])
            self.l1_penalty_tf = self.lbd * tf.reduce_sum(tf.abs(
                tf.boolean_mask(self.Ws, self.causality_matrix==0)))

        # Define the unnormalized loss function.
        self.loss_tf = -sharpe_tf(self.positions_tf, self.batch_out_tf,
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

    def shuffle(self, batch_in):
        """ Permute the batch's feature using a fixed permutation.

        Args:
          batch_in (n_batch, horizon, n_ftrs): Numpy array.

        Returns
          batch_in_perm (n_batch, n_time, n_ftrs): Same array, but
            all the time indices and feature indices are permuted
            randomly, *except* the last time index.
        """
        n_batch = tf.shape(batch_in)[0]
        lasttime = batch_in[:, -1:, :]
        flatothers = tf.reshape(batch_in[:, :-1, :],
                                (n_batch, (self.horizon - 1) *
                                 self.n_ftrs))
        # Permute flatothers.
        flatothers = tf.transpose(tf.gather(tf.transpose(flatothers),
                                            self.data_permutation))
        flatothers = tf.reshape(flatothers, (n_batch, self.horizon - 1,
                                             self.n_ftrs))
        return tf.concat([flatothers, lasttime], 1)

    def restart_variables(self):
        """ Reinitialize the weight and bias matrices. """
        self.sess.run(self.init_op)

    def get_weights(self):
        """ Get the (numpy) weight matrix. """
        return self.sess.run(self.Ws)

    def _positions_np(self, batch_in):
        """ Predict a portfolio for a training batch.

        Args:
          batch_in (n_batch, n_time, n_ftrs): Input data.

        Returns:
          positions (n_batch, n_markets): Positions.
        """
        return self.sess.run(self.positions_tf,
                             {self.batch_in_tf: batch_in})

    def predict(self, data_in):
        """ Predict a portfolio for a test batch.

        Args:
          data_in (horizon, n_ftrs): Input data, where
            horizon = n_time - n_sharpe + 1. This corresponds
            to data needed to predict just one portfolio.

        Returns:
          positions (n_markets): Positions.
        """
        data_in = np.vstack([np.zeros((self.n_sharpe-1, self.n_ftrs)),
                             data_in])
        data_in = np.expand_dims(data_in, axis=0)
        # Pad the data with n_sharpe-1 fake datapoints.
        return self.sess.run(self.positions_tf,
                             {self.batch_in_tf: data_in})[-1, -1]

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

    def save(self, fname='saved_data/chunkmodel'):
        self.save_path = self.saver.save(self.sess, fname)

    def load(self):
        self.saver.restore(self.sess, self.save_path)
