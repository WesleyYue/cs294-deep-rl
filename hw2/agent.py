import enum
import inspect
import logging
import multiprocessing as mp
import os
import time

import gym
import numpy as np
import tensorflow as tf

import logz
from model import PolicyGradient

logger = logging.getLogger(__name__)


def _agent_debug(debug_msg):
    logger.debug("Agent" + str(os.getpid()) + debug_msg)


class Agent(mp.Process):
    @enum.unique
    class States(enum.Enum):
        ROLLOUT = enum.auto()
        TRAIN = enum.auto()
        TERMINATE = enum.auto()
        UPDATE = enum.auto()  # update policy

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
            state=None,
            network_weights=None,
            paths_queue=None,
            num_agents=1):  # TODO(wy): fix this constructor argument mess

        mp.Process.__init__(self)

        self.network_parameters = network_parameters
        self.env_name = env_name
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch / num_agents

        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.nn_baseline = nn_baseline
        self.normalize_advantages = normalize_advantages

        # Shared IPC Queues
        self.results = results
        self.state_queue = state
        self.network_weights = network_weights
        self.paths_queue = paths_queue

        # TODO(wy): num_agents is passed in only for the benefit of being able
        # to assert that the shared results queue is empty after looping through
        # num_agents, during training state. Probably better to just remove this
        # parameter and loop until results empty.
        self.num_agents = num_agents
        self.env = None  # Gym environment

        tf.set_random_seed(seed)
        np.random.seed(seed)

        self._model = None

    def _simulate(self):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        enough_timesteps = False
        while not enough_timesteps:
            ob = self.env.reset()
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
                ob, rew, done, _ = self.env.step(ac)
                rewards.append(rew)
                steps += 1

                done_trajectory = (done or steps > self.max_path_length)

            path = {
                "observation": np.array(obs),
                "reward": np.array(rewards),
                "action": np.array(acs)
            }
            paths.append(path)
            timesteps_this_batch += len(path['reward'])

            enough_timesteps = (timesteps_this_batch >
                                self.min_timesteps_per_batch)
        return paths, timesteps_this_batch

    def _rollout(self):
        total_timesteps = 0

        paths, timesteps_this_batch = self._simulate()

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

        # The observations, actions, and advantages are put in the same queue
        # instead of their own separate queues to ensure the correct
        # observations are mapped to the correction actions and advantages.
        # Implementing these in separate multiprocessing.Queue's would not be
        # able to guarantee the ordering.
        self.results.put({
            "observations": ob_no,
            "actions": ac_na,
            "advantages": adv_n
        })
        self.paths_queue.put(paths)

    def run(self):
        # Remapped so tests don't need to actually spawn new processes
        self._main()

    def _setup(self):
        """Initial setup code when process first spawns."""

        self.env = gym.make(self.env_name)

        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        # Maximum length for episodes
        self.max_path_length = self.max_path_length or self.env.spec.max_episode_steps

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]

        self.model = PolicyGradient(
            ob_dim, ac_dim, discrete, self.network_parameters["n_layers"],
            self.network_parameters["size"],
            self.network_parameters["learning_rate"], self.nn_baseline)

    def _main(self):
        self._setup()

        prev_state = None
        while True:
            # _agent_debug(" getting new task.")
            state = self.state_queue.get()

            if state is Agent.States.ROLLOUT:
                _agent_debug(".ROLLOUT")
                self._rollout()


                # Block the system state from transitioning until all agents
                # have picked up a ROLLOUT task. This prevents an agent that has
                # picked up a ROLLOUT task from picking up a second ROLLOUT task
                # until the agent transitions to another state.
                while not self.state_queue.empty():
                    pass
                self.state_queue.task_done()
                prev_state = Agent.States.ROLLOUT

            elif state is Agent.States.TRAIN:
                _agent_debug(".TRAIN")

                # Each agent puts their own result object on the queue and they
                # need to be consolidated before training.
                observations = []
                actions = []
                advantages = []
                for _ in range(self.num_agents):
                    results = self.results.get()
                    observations.extend(results["observations"])
                    actions.extend(results["actions"])
                    advantages.extend(results["advantages"])

                assert self.results.empty()

                weights = self.model.train(observations, actions, advantages)

                # Not ideal b/c need to put duplicate sets of weights on queue
                # for each agent. TODO(wy)
                for _ in range(self.num_agents):
                    self.network_weights.put(weights)
                self.state_queue.task_done()
                prev_state = Agent.States.TRAIN

            elif state is Agent.States.UPDATE:
                # TODO(wy): Implement logic to not update if there is only one
                # thread since the training should have already updated the
                # weights simultanenously
                _agent_debug(".UDPATE")

                # Check that agent is not taking two UPDATE tasks in one epoch
                assert prev_state is not Agent.States.UPDATE

                self._load_weights(self.network_weights.get())

                # Hack to reduce the probability that the same agent picks up
                # UPDATE again, before another agent who was supposed to be
                # updated got a chance.
                time.sleep(0.1)

                self.state_queue.task_done()
                prev_state = Agent.States.UPDATE

            elif state is Agent.States.TERMINATE:
                _agent_debug(".TERMINATE")
                self.state_queue.task_done()
                assert self.results.empty()
                assert self.network_weights.empty()
                assert self.paths_queue.empty()
                prev_state = Agent.States.TERMINATE
                break

            else:
                raise ValueError(
                    "Agent state queue was not one of ROLLOUT or TRAIN")

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

