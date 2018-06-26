import logging
import time
from pprint import pprint

import tensorflow as tf

logger = logging.getLogger(__name__)


def build_mlp(input_placeholder,
              output_size,
              scope,
              n_layers=2,
              size=64,
              activation=tf.nn.relu,
              output_activation=None):
    #==========================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a
    # multilayer perceptron) with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation
    # 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #==========================================================================#

    with tf.variable_scope(scope):
        hidden_layers = [
            tf.layers.dense(input_placeholder, size, activation=activation)
        ]
        for layer_n in range(1, n_layers):
            hidden_layers.append(
                tf.layers.dense(
                    hidden_layers[layer_n - 1], size, activation=activation))
        output = tf.layers.dense(
            hidden_layers[n_layers - 1],
            output_size,
            activation=output_activation)
        return output


class PolicyGradient:
    def __init__(self, ob_dim, ac_dim, discrete, n_layers, size, learning_rate,
                 nn_baseline):
        """Implementation of policy gradient neural network, with ability to load/dump weights.

        Arguments:
            ob_dim {integer} -- Observation space dimensions
            ac_dim {integer} -- Action space dimensions
            discrete {boolean} -- Whether the actions are discrete
            n_layers {integer} -- Number of layers in the multi-layer perceptron
            size {integer} -- Size of each hidden layer in the MLP
            learning_rate {float} -- Learning rate of the MLP
            nn_baseline {boolean} -- Whether to use a neural network baseline
        """

        #========================================================================================#
        #                           ----------SECTION 4----------
        # Placeholders
        #
        # Need these for batch observations / actions / advantages in policy gradient loss function.
        #========================================================================================#

        self.sy_ob_no = tf.placeholder(
            shape=[None, ob_dim], name="ob", dtype=tf.float32)
        if discrete:
            self.sy_ac_na = tf.placeholder(
                shape=[None], name="ac", dtype=tf.int32)
        else:
            self.sy_ac_na = tf.placeholder(
                shape=[None, ac_dim], name="ac", dtype=tf.float32)

        # Define a placeholder for advantages
        self.sy_adv_n = tf.placeholder(
            shape=[None], name="adv", dtype=tf.float32)

        #========================================================================================#
        #                           ----------SECTION 4----------
        # Networks
        #
        # Make symbolic operations for
        #   1. Policy network outputs which describe the policy distribution.
        #       a. For the discrete case, just logits for each action.
        #
        #       b. For the continuous case, the mean / log std of a Gaussian distribution over
        #          actions.
        #
        #      Hint: use the 'build_mlp' function you defined in utilities.
        #
        #      Note: these ops should be functions of the placeholder 'self.sy_ob_no'
        #
        #   2. Producing samples stochastically from the policy distribution.
        #       a. For the discrete case, an op that takes in logits and produces actions.
        #
        #          Should have shape [None]
        #
        #       b. For the continuous case, use the reparameterization trick:
        #          The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        #
        #               mu + sigma * z,         z ~ N(0, I)
        #
        #          This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        #
        #          Should have shape [None, ac_dim]
        #
        #      Note: these ops should be functions of the policy network output ops.
        #
        #   3. Computing the log probability of a set of actions that were actually taken,
        #      according to the policy.
        #
        #      Note: these ops should be functions of the placeholder 'self.sy_ac_na', and the
        #      policy network output ops.
        #
        #========================================================================================#

        if discrete:
            sy_logits_na = build_mlp(self.sy_ob_no, ac_dim, scope="discrete")
            # Hint: Use the tf.multinomial op
            self.sy_sampled_ac = tf.squeeze(
                tf.multinomial(sy_logits_na, 1), axis=1)
            sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.sy_ac_na, logits=sy_logits_na)

        else:
            sy_mean = build_mlp(
                self.sy_ob_no,
                ac_dim,
                scope="continuous",
                n_layers=n_layers,
                size=size)
            # logstd should just be a trainable variable, not a network output.
            sy_logstd = tf.get_variable("logstd", [ac_dim])
            sy_std = tf.exp(sy_logstd)
            self.sy_sampled_ac = sy_mean + sy_std * \
                tf.random_normal(tf.shape(sy_mean))
            # Hint: Use the log probability under a multivariate gaussian.
            sy_logprob_n = 0.5 * \
                tf.reduce_sum(tf.square(sy_mean - self.sy_ac_na) / sy_std, 1)

        # Loss function that we'll differentiate to get the policy gradient.
        loss = tf.reduce_mean(sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #======================================================================#
        #                           ----------SECTION 5----------
        # Optional Baseline
        #======================================================================#

        if nn_baseline:
            self.baseline_prediction = tf.squeeze(
                build_mlp(
                    self.sy_ob_no,
                    1,
                    "nn_baseline",
                    n_layers=n_layers,
                    size=size))
            # Define placeholders for targets, a loss function and an update op
            # for fitting a neural network baseline. These will be used to fit
            # the neural network baseline.

            self.sy_q_n = tf.placeholder(
                shape=[None], name="q", dtype=tf.float32)
            baseline_loss = tf.losses.mean_squared_error(
                self.sy_q_n, self.baseline_prediction)
            self.baseline_update_op = tf.train.AdamOptimizer(
                learning_rate).minimize(baseline_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run(self, observations):
        return self.sess.run(
            self.sy_sampled_ac, feed_dict={self.sy_ob_no: observations[None]})

    def predict_baseline(self, observations):
        # TODO(wy): Breakout baseline prediction as its own class
        # Output distribution should already be normalized from training
        return self.sess.run(
            self.baseline_prediction,
            feed_dict={self.sy_ob_no: observations})

    def train_baseline(self, observations, baseline_predictions,
                       normalized_q_n):
        self.sess.run(
            self.baseline_update_op,
            feed_dict={
                self.sy_ob_no: observations,
                self.sy_q_n: normalized_q_n,
                self.baseline_prediction: baseline_predictions
            })

    def train(self, observations, actions, advantages):
        logger.debug("Training...")

        self.sess.run(
            self.update_op,
            feed_dict={
                self.sy_ob_no: observations,
                self.sy_ac_na: actions,
                self.sy_adv_n: advantages
            })

        return self.dump_weights()

    def dump_weights(self):
        logger.debug("Dumping weights...")
        weights = {}
        for v in tf.trainable_variables():
            weights[v.name] = self.sess.run(v)
        return weights

    def load_weights(self, weights):
        logger.debug("Loading weights...")
        for v in tf.trainable_variables():
            self.sess.run(tf.assign(v, weights[v.name]))

        assert len(weights) == len(tf.trainable_variables())
