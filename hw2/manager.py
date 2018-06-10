import enum
import inspect
import multiprocessing as mp
import time

import gym
import numpy as np
import tensorflow as tf

import logz
from model import PolicyGradient
from agent import Agent

#region notation definition
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
#endregion


class Manager(mp.Process):
    @enum.unique
    class AgentStates(enum.Enum):
        ROLLOUT = enum.auto()
        WAITING_FOR_PARAMETERS = enum.auto()
        READY_TO_ROLLOUT = enum.auto()

    def __init__(
            self,
            exp_name='',  # Not used. Only for the logz convienence
            env_name='CartPole-v0',
            epoches=100,
            gamma=1.0,
            min_timesteps_per_batch=1000,
            max_path_length=None,
            learning_rate=5e-3,
            reward_to_go=True,
            #  animate=True,
            logdir=None,
            normalize_advantages=True,
            nn_baseline=False,
            seed=0,
            # network arguments
            n_layers=1,
            size=32,
            num_agents=0):

        if num_agents < 1:
            raise ValueError("Need at least 1 agent to do the rollout")

        mp.Process.__init__(self)

        self.network_parameters = {
            "n_layers": n_layers,
            "size": size,
            "learning_rate": learning_rate
        }

        self.epoches = epoches
        self.env_name = env_name
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch
        self.seed = seed

        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.nn_baseline = nn_baseline
        self.normalize_advantages = normalize_advantages

        self.state = self.AgentStates.WAITING_FOR_PARAMETERS

        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Configure logz
        logz.configure_output_dir(logdir)
        args = inspect.getargspec(Agent.__init__)[0]
        locals_ = locals()
        # args[1:] to skip over the 'self' parameter
        params = {k: locals_[k] if k in locals_ else None for k in args[1:]}
        logz.save_params(params)

    def run(self):
        start = time.time()

        env = gym.make(self.env_name)

        discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # Observation and action sizes
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

        for itr in range(self.epoches):
            print("********** Epoch %i ************" % itr)

            # Initialize students processes that will perform the rollouts
            results = mp.Queue()
            agent_state = mp.Value("i")

            # 1 = rollout
            # 2 = train
            # 3 = load weights

            self.agents = [
                Agent(self.env_name, self.gamma, self.min_timesteps_per_batch,
                      self.max_path_length, self.reward_to_go,
                      self.normalize_advantages, self.nn_baseline, self.seed,
                      self.network_parameters, results, agent_state)
            ]

            self.agents[0].start()

            result = results.get()
            # weights = self.agents[0].train(
            #     result["observations"], result["actions"], result["advantage"])
            
            # for agent in self.agents:
            #     agent.load_weights(weights)

            self.agents[0].join()

            # self.model.train(result["observations"], result["actions"], result["advantage"])

            # Log diagnostics
            returns = [path["reward"].sum() for path in result["paths"]]
            ep_lengths = [len(path["reward"]) for path in result["paths"]]
            logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("Iteration", itr)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            # logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
            # logz.log_tabular("TimestepsSoFar", total_timesteps)
            logz.dump_tabular()
            # logz.pickle_tf_vars()
