import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
import multiprocessing as mp #TODO(wy) clean up
from multiprocessing import Process
from model import PolicyGradient
from distributed import Agent


#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32
             ):

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]


    mp.set_start_method('spawn')
    agent = Agent(n_iter, env_name, max_path_length,
                  logdir, min_timesteps_per_batch, gamma, reward_to_go, nn_baseline, normalize_advantages, ob_dim, ac_dim, discrete, n_layers, size,
                  learning_rate)
    agent.start()
    agent.join()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        # def train_func():
        #     train_PG(
        #         exp_name=args.exp_name,
        #         env_name=args.env_name,
        #         n_iter=args.n_iter,
        #         gamma=args.discount,
        #         min_timesteps_per_batch=args.batch_size,
        #         max_path_length=max_path_length,
        #         learning_rate=args.learning_rate,
        #         reward_to_go=args.reward_to_go,
        #         animate=args.render,
        #         logdir=os.path.join(logdir,'%d'%seed),
        #         normalize_advantages=not(args.dont_normalize_advantages),
        #         nn_baseline=args.nn_baseline, 
        #         seed=seed,
        #         n_layers=args.n_layers,
        #         size=args.size
        #         )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        # p = Process(target=train_func, args=tuple())
        # p.start()
        # p.join()

        train_PG(
            exp_name=args.exp_name,
            env_name=args.env_name,
            n_iter=args.n_iter,
            gamma=args.discount,
            min_timesteps_per_batch=args.batch_size,
            max_path_length=max_path_length,
            learning_rate=args.learning_rate,
            reward_to_go=args.reward_to_go,
            animate=args.render,
            logdir=os.path.join(logdir,'%d'%seed),
            normalize_advantages=not(args.dont_normalize_advantages),
            nn_baseline=args.nn_baseline,
            seed=seed,
            n_layers=args.n_layers,
            size=args.size
            )
        

if __name__ == "__main__":
    main()

    ###

    # sy_ob_no = tf.placeholder(
    #     shape=[None, 24], name="ob", dtype=tf.float32)
    # build_mlp(sy_ob_no, 33, "test24", n_layers=5, size= 35)

    ###

    # with tf.Session() as sess:

    #     placehold = tf.placeholder(
    #         shape=[1, 2], name='input', dtype=tf.float32)

    #     mlp = build_mlp(
    #         input_placeholder=placehold,
    #         output_size=10,
    #         scope="alpha"
    #     )

    #     merge = tf.summary.merge_all()

    #     tf.global_variables_initializer().run()  # pylint: disable=E1101
    #     summary, _ = sess.run([merge, mlp], feed_dict={placehold: [[1, 2]]})

    #     train_writer = tf.summary.FileWriter('./logs/alpha', sess.graph)
    #     train_writer.add_summary(summary)
