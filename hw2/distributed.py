from multiprocessing import Process
from model import PolicyGradient
import time
import gym
import numpy as np
import logz
import inspect
import tensorflow as tf
import enum

# class AgentManager(Process):
#     def __init__(self, ob_dim, ac_dim, discrete, n_layers, size, learning_rate,
#             nn_baseline, num_agents):
#         Process.__init__(self)
#         # self.ob_dim = ob_dim
#         # self.ac_dim = ac_dim
#         # self.discrete = discrete
#         # self.n_layers = n_layers
#         # self.size = size
#         # self.learning_rate = learning_rate
#         # self.nn_baseline = nn_baseline

#         self.agents = [Agent(
#             Model(ob_dim, ac_dim, discrete, n_layers, size, learning_rate, nn_baseline))]

#     def run(self):
#         # Set up tensorflow models


#         while True:

#             if :
#                 break

#     def update(self):
#         pass

class Agent(Process):
    @enum.unique
    class AgentStates(enum.Enum):
        ROLLOUT = enum.auto()
        WAITING_FOR_PARAMETERS = enum.auto()
        READY_TO_ROLLOUT = enum.auto()

    def __init__(self,
                 exp_name='',  # Not used. Only for the logz convienence
                 env_name='CartPole-v0',
                 n_iter=100,
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
                 size=32):
        Process.__init__(self)

        self.network_parameters = {
            "n_layers": n_layers,
            "size": size,
            "learning_rate": learning_rate
        }

        self.n_iter = n_iter
        self.env_name = env_name
        self.max_path_length = max_path_length
        self.min_timesteps_per_batch = min_timesteps_per_batch

        self.gamma = gamma
        self.reward_to_go = reward_to_go
        self.nn_baseline = nn_baseline
        self.normalize_advantages = normalize_advantages

        self.state = self.AgentStates.WAITING_FOR_PARAMETERS

        # Set random seeds
        tf.set_random_seed(seed)
        np.random.seed(seed)

        # Configure output directory for logging
        logz.configure_output_dir(logdir)

        # Log experimental parameters
        args = inspect.getargspec(Agent.__init__)[0]
        locals_ = locals()
        # args[1:] to skip over the 'self' parameter
        params = {k: locals_[k] if k in locals_ else None for k in args[1:]}
        logz.save_params(params)

    # def _run_init(self):
    #     """Initialization tasks that run in the spawned process instead of the
    #     process that is instantiating the class."""

    def rollout(self, env, max_path_length):
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
        start = time.time()

        env = gym.make(self.env_name)

        # Is this env continuous, or discrete?
        discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # Maximum length for episodes
        max_path_length = self.max_path_length or env.spec.max_episode_steps

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

        self.model = PolicyGradient(ob_dim,
                                    ac_dim,
                                    discrete,
                                    self.network_parameters["n_layers"],
                                    self.network_parameters["size"],
                                    self.network_parameters["learning_rate"],
                                    self.nn_baseline)

        #========================================================================================#
        # Training Loop
        #========================================================================================#

        total_timesteps = 0

        for itr in range(self.n_iter):
            print("********** Iteration %i ************" % itr)

            paths, timesteps_this_batch = self.rollout(env, max_path_length)

            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient update by concatenating
            # across paths
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])

            #====================================================================================#
            #                           ----------SECTION 4----------
            # Computing Q-values
            #
            # Your code should construct numpy arrays for Q-values which will be used to compute
            # advantages (which will in turn be fed to the placeholder you defined above).
            #
            # Recall that the expression for the policy gradient PG is
            #
            #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
            #
            # where
            #
            #       tau=(s_0, a_0, ...) is a trajectory,
            #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
            #       and b_t is a baseline which may depend on s_t.
            #
            # You will write code for two cases, controlled by the flag 'reward_to_go':
            #
            #   Case 1: trajectory-based PG
            #
            #       (reward_to_go = False)
            #
            #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over
            #       entire trajectory (regardless of which time step the Q-value should be for).
            #
            #       For this case, the policy gradient estimator is
            #
            #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
            #
            #       where
            #
            #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
            #
            #       Thus, you should compute
            #
            #           Q_t = Ret(tau)
            #
            #   Case 2: reward-to-go PG
            #
            #       (reward_to_go = True)
            #
            #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
            #       from time step t. Thus, you should compute
            #
            #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            #
            #
            # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            # like the 'ob_no' and 'ac_na' above.
            #
            #====================================================================================#

            # YOUR_CODE_HERE
            q_n = []

            for path in paths:
                q = 0
                q_trajectory = []

                # Calculate reward to go with gamma
                for reward in reversed(path["reward"]):
                    q = reward + q * self.gamma
                    q_trajectory.append(q)
                q_trajectory.reverse()

                if not self.reward_to_go:
                    # Replace all Qt with Q0
                    q_trajectory = [q_trajectory[0]] * len(q_trajectory)

                q_n.extend(q_trajectory)

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
                # YOUR_CODE_HERE
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

            # YOUR_CODE_HERE
            # _, loss_value = sess.run([update_op, loss], feed_dict={sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
            # print(loss_value)
            self.model.train(ob_no, ac_na, adv_n)

            # Log diagnostics
            returns = [path["reward"].sum() for path in paths]
            ep_lengths = [len(path["reward"]) for path in paths]
            logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("Iteration", itr)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            logz.dump_tabular()
            # logz.pickle_tf_vars()
