import tensorflow as tf
import time

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.nn.relu,
        output_activation=None):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units.
    #
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#

    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        hidden_layers = [tf.layers.dense(
            input_placeholder, size, activation=activation)]
        for layer_n in range(1, n_layers):
            hidden_layers.append(tf.layers.dense(
                hidden_layers[layer_n - 1], size, activation=activation))
        output = tf.layers.dense(
            hidden_layers[n_layers - 1], output_size, activation=output_activation)
        return output

class PolicyGradient:
    def __init__(self, ob_dim, ac_dim, discrete, n_layers, size, learning_rate, nn_baseline):

        #========================================================================================#
        #                           ----------SECTION 4----------
        # Placeholders
        #
        # Need these for batch observations / actions / advantages in policy gradient loss function.
        #========================================================================================#

        self.sy_ob_no = tf.placeholder(
            shape=[None, ob_dim], name="ob", dtype=tf.float32)
        if discrete:
            self.sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            self.sy_ac_na = tf.placeholder(
                shape=[None, ac_dim], name="ac", dtype=tf.float32)

        # Define a placeholder for advantages
        self.sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

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
            self.sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=1)
            sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.sy_ac_na, logits=sy_logits_na)

        else:
            sy_mean = build_mlp(
                self.sy_ob_no, ac_dim, scope="continuous", n_layers=n_layers, size=size)
            # logstd should just be a trainable variable, not a network output.
            sy_logstd = tf.get_variable("logstd", [ac_dim])
            sy_std = tf.exp(sy_logstd)
            self.sy_sampled_ac = sy_mean + sy_std * tf.random_normal(tf.shape(sy_mean))
            # Hint: Use the log probability under a multivariate gaussian.
            sy_logprob_n = 0.5 * \
                tf.reduce_sum(tf.square(sy_mean - self.sy_ac_na) / sy_std, 1)

        # Loss function that we'll differentiate to get the policy gradient.
        loss = tf.reduce_mean(sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        #========================================================================================#
        #                           ----------SECTION 5----------
        # Optional Baseline
        #========================================================================================#

        if nn_baseline:
            baseline_prediction = tf.squeeze(build_mlp(
                self.sy_ob_no,
                1,
                "nn_baseline",
                n_layers=n_layers,
                size=size))
            # Define placeholders for targets, a loss function and an update op for fitting a
            # neural network baseline. These will be used to fit the neural network baseline.
            # YOUR_CODE_HERE
            # baseline_update_op = TODO


        #========================================================================================#
        # Tensorflow Engineering: Config, Session, Variable initialization
        #========================================================================================#

        tf_config = tf.ConfigProto(
            inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

        self.sess = tf.Session(config=tf_config)
        # self.sess.__enter__()  # equivalent to `with sess:`
        self.sess.run(tf.global_variables_initializer())
        # tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101
    
    def run(self, observations):
        # print("Running...")
        # if hasattr(self, "var"):
        #     self.load_weights(self.var)

        # for v in tf.trainable_variables():
        #     print(v.name)
        #     print(type(self.sess.run(v)))
        # time.sleep(10)
        return self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: observations[None]})

    def train(self, observations, actions, advantages):
        print("Training...")

        # print(self.sess.run("continuous/dense/kernel:0"))
        self.sess.run(self.update_op, 
                      feed_dict={
                          self.sy_ob_no: observations, 
                          self.sy_ac_na: actions, 
                          self.sy_adv_n: advantages})
        # print(self.sess.run("continuous/dense/kernel:0"))
        # time.sleep(10)

        # self.var = self.dump_weights()

    def dump_weights(self):
        print("Dumping weights...")
        var = {}
        for v in tf.trainable_variables():
            var[v.name] = v
        return var
    
    def load_weights(self, var):
        # print("Loading weights...")
        for v in tf.trainable_variables():
            v = var[v.name]

        assert len(var) == len(tf.trainable_variables())
