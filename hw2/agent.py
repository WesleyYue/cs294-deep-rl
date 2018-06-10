import enum
import inspect
import multiprocessing as mp
import time

import gym
import numpy as np
import tensorflow as tf

import logz
from model import PolicyGradient


class Agent(mp.Process):
    @enum.unique
    class AgentStates(enum.Enum):
        ROLLOUT = enum.auto()
        WAITING_FOR_PARAMETERS = enum.auto()
        READY_TO_ROLLOUT = enum.auto()

    def __init__(
            self,
            env_name='CartPole-v0',
            gamma=1.0,
            min_timesteps_per_batch=1000,  # timesteps to rollout before reporting observations
            max_path_length=None,
            reward_to_go=True,
            normalize_advantages=True,
            nn_baseline=False,
            seed=0,
            network_parameters={
                "n_layers": 1,
                "size": 32,
                "learning_rate": 5e-3
            },
            results=None,
            state=None):

        mp.Process.__init__(self)

        self.network_parameters = network_parameters
        self.env_name = env_name
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch

        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.nn_baseline = nn_baseline
        self.normalize_advantages = normalize_advantages

        self.results = results

        self.state = state

        tf.set_random_seed(seed)
        np.random.seed(seed)


    def _rollout(self, env, max_path_length):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        enough_timesteps = False
        while not enough_timesteps:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            # animate_this_episode = (
            #     len(paths) == 0 and (itr % 10 == 0) and animate)
            steps = 0

            done_trajectory = False
            while not done_trajectory:
                # if animate_this_episode:
                #     env.render()
                #     time.sleep(0.05)
                obs.append(ob)
                ac = self.model.run(ob)
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1

                done_trajectory = (done or steps > max_path_length)

            path = {"observation": np.array(obs),
                    "reward": np.array(rewards),
                    "action": np.array(acs)}
            paths.append(path)
            timesteps_this_batch += len(path['reward'])

            enough_timesteps = (timesteps_this_batch >
                                self.min_timesteps_per_batch)
        return paths, timesteps_this_batch


    def run(self):
        env = gym.make(self.env_name)

        discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # Maximum length for episodes
        max_path_length = self.max_path_length or env.spec.max_episode_steps

        # Observation and action sizes
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

        self.model = PolicyGradient(ob_dim,
                                    ac_dim,
                                    discrete,
                                    self.network_parameters["n_layers"],
                                    self.network_parameters["size"],
                                    self.network_parameters["learning_rate"],
                                    self.nn_baseline,
                                    "agent")  # TODO(wy): this needs to be numbered when there are more than 1 agents
        print("done setting up model")

        total_timesteps = 0

        paths, timesteps_this_batch = self._rollout(env, max_path_length)

        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])

        q_n = self._compute_q_values(self.gamma, paths, self.reward_to_go)

        #region nn_baseline and normalize_advantage
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)

            # b_n = TODO
            # adv_n = q_n - b_n

            adv_n = q_n.copy()  # TODO(wesley): remove this
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)

        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if self.nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the
            # baseline.
            #
            # Fit it to the current batch in order to use for the next iteration. Use the
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            pass


        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on
        # the current batch of rollouts.
        #
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below.
        #endregion

        self.results.put({"observations": ob_no, "actions": ac_na, "advantage": adv_n, "paths": paths})


    @staticmethod
    def _compute_q_values(gamma, paths, reward_to_go):
        q_n = []

        for path in paths:
            q = 0
            q_trajectory = []

            # Calculate reward to go with gamma
            for reward in reversed(path["reward"]):
                q = reward + q * gamma
                q_trajectory.append(q)
            q_trajectory.reverse()

            if not reward_to_go:
                # All q is same as the first q (reward of full trajectory from
                # beginning) when not doing reward to go
                q_trajectory = [q_trajectory[0]] * len(q_trajectory)

            q_n.extend(q_trajectory)

        return q_n

    def _train(self, observations, actions, advantages):
        return self.model.train(observations, actions, advantages)

    def _load_weights(self, weights):
        self.model.load_weights(weights)
