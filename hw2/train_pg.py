import inspect
import multiprocessing
import os
import time

import gym
import numpy as np
import scipy.signal
import tensorflow as tf

import logz
from agent import Agent
from manager import Manager

#============================================================================================#
# Policy Gradient
#============================================================================================#

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
    parser.add_argument('--dont_normalize_advantages',
                        '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + \
        '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)

        manager = Manager(exp_name=args.exp_name,
                      env_name=args.env_name,
                      epoches=args.n_iter,
                      gamma=args.discount,
                      min_timesteps_per_batch=args.batch_size,
                      max_path_length=max_path_length,
                      learning_rate=args.learning_rate,
                      reward_to_go=args.reward_to_go,
                    #   animate=args.render,
                      logdir=os.path.join(logdir, '%d' % seed),
                      normalize_advantages=not(args.dont_normalize_advantages),
                      nn_baseline=args.nn_baseline,
                      seed=seed,
                      n_layers=args.n_layers,
                      size=args.size,
                      num_agents=1)
        manager.start()
        manager.join()


if __name__ == "__main__":
    main()
