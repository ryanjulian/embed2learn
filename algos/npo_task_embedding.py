import time

import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import ext
from garage.misc import special
from garage.misc.overrides import overrides
import garage.misc.logger as logger

from garage.tf.algos import BatchPolopt
from garage.tf.core import JointParameterized
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FirstOrderOptimizer

from sandbox.embed2learn.algos.utils import flatten_batch
from sandbox.embed2learn.algos.utils import flatten_batch_dict
from sandbox.embed2learn.algos.utils import filter_valids
from sandbox.embed2learn.algos.utils import filter_valids_dict
from sandbox.embed2learn.embeddings import GaussianMLPMultitaskPolicy
from sandbox.embed2learn.embeddings import StochasticMultitaskPolicy
from sandbox.embed2learn.embeddings import StochasticEmbedding
from sandbox.embed2learn.samplers import TaskEmbeddingSampler
from sandbox.embed2learn.samplers.task_embedding_sampler import rollout


def _optimizer_or_default(optimizer, args):
    use_optimizer = optimizer
    use_args = args
    if use_optimizer is None:
        if use_args is None:
            use_args = dict()
        use_optimizer = ConjugateGradientOptimizer(**use_args)
    return use_optimizer


class NPOTaskEmbedding(BatchPolopt, Serializable):
    """
    Natural Policy Optimization with Task Embeddings
    """

    def __init__(self,
                 name="NPOTaskEmbedding",
                 plot_warmup_itrs=0,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.01,
                 policy_ent_coeff=1e-2,
                 policy=None,
                 task_encoder_ent_coeff=1e-5,
                 trajectory_encoder=None,
                 trajectory_encoder_optimizer=None,
                 trajectory_encoder_optimizer_args=None,
                 trajectory_encoder_ent_coeff=1e-3,
                 trajectory_encoder_learning_rate=1e-3,
                 **kwargs):
        Serializable.quick_init(self, locals())
        assert kwargs['env'].task_space
        assert isinstance(policy, StochasticMultitaskPolicy)
        assert isinstance(trajectory_encoder, StochasticEmbedding)

        with tf.variable_scope(name):
            self.name = name
            self.plot_warmup_itrs = plot_warmup_itrs

            # Optimizer for policy + task encoder
            self.optimizer = _optimizer_or_default(optimizer, optimizer_args)
            self.step_size = float(step_size)
            self.policy_ent_coeff = float(policy_ent_coeff)
            self.task_enc_ent_coeff = float(task_encoder_ent_coeff)

            self.traj_encoder = trajectory_encoder
            self.traj_enc_ent_coeff = trajectory_encoder_ent_coeff
            self.traj_enc_optimizer = FirstOrderOptimizer(
                tf.train.AdamOptimizer,
                dict(learning_rate=float(trajectory_encoder_learning_rate)))

            self.z_summary = None
            self.z_summary_op = None

            sampler_cls = TaskEmbeddingSampler
            sampler_args = dict(trajectory_encoder=self.traj_encoder, )
            super(NPOTaskEmbedding, self).__init__(
                sampler_cls=sampler_cls,
                sampler_args=sampler_args,
                policy=policy,
                **kwargs)

    @overrides
    def init_opt(self):
        pol_loss, pol_mean_kl, input_list, traj_enc_loss, traj_enc_inputs = \
            self._build_opt()

        # Optimize policy (with embedding) and traj_encoder jointly
        pol_embed = JointParameterized(
            components=[self.policy, self.policy.embedding])

        self.optimizer.update_opt(
            loss=pol_loss,
            target=pol_embed,
            leq_constraint=(pol_mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl")

        # Optimize trajectory encoder separately via supervised learning
        self.traj_enc_optimizer.update_opt(
            loss=traj_enc_loss,
            target=self.traj_encoder,
            inputs=traj_enc_inputs)

        return dict()

    def _build_opt(self):
        with tensor_utils.enclosing_scope(self.name, "build_opt"):

            is_recurrent = int(self.policy.recurrent)
            if is_recurrent:
                raise NotImplementedError

            #### Policy and loss function #####################################
            # Input variables
            self._obs_var = self.policy.observation_space.new_tensor_variable(
                'obs',
                extra_dims=1 + 1,
            )
            self._task_var = self.policy.task_space.new_tensor_variable(
                'task',
                extra_dims=1 + 1,
            )
            self._action_var = self.env.action_space.new_tensor_variable(
                'action',
                extra_dims=1 + 1,
            )
            self._reward_var = tensor_utils.new_tensor(
                'reward',
                ndim=1 + 1,
                dtype=tf.float32,
            )
            self._latent_var = self.policy.latent_space.new_tensor_variable(
                'latent',
                extra_dims=1 + 1,
            )
            self._baseline_var = tensor_utils.new_tensor(
                'baseline',
                ndim=1 + 1,
                dtype=tf.float32,
            )
            self._trajectory_var = self.traj_encoder.input_space.new_tensor_variable(
                'trajectory',
                extra_dims=1 + 1,
            )
            self._valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="valid")

            self.initialize_vars()

            policy_input_list = self.build_policy_input()
            surr_loss, pol_mean_kl, traj_enc_loss, rewards = self.build_loss()
            returns = self.build_returns(rewards)

            # Outputs
            self._traj_loss = traj_enc_loss
            self._pol_mean_kl = pol_mean_kl
            self._surr_loss = surr_loss

            # Functions
            self.f_rewards = tensor_utils.compile_function(
                policy_input_list, rewards, log_name="f_rewards")
            self.f_returns = tensor_utils.compile_function(
                policy_input_list, returns, log_name="f_returns")
            self.f_task_entropies = tensor_utils.compile_function(
                policy_input_list,
                self._all_task_entropies,
                log_name="f_task_entropies")
            self.f_policy_entropy = tensor_utils.compile_function(
                policy_input_list,
                tf.reduce_sum(self._pol_entropy * self._valid_var),
                log_name="f_policy_entropy")
            self.f_traj_cross_entropy = tensor_utils.compile_function(
                policy_input_list,
                tf.reduce_sum(self._traj_ll * self._valid_var),
                log_name="f_traj_cross_entropy")

            traj_enc_input_list = self.build_traj_enc_input()

            return self._surr_loss, self._pol_mean_kl, policy_input_list, self._traj_loss, traj_enc_input_list

    ############################ initialize base variables ####################
    def initialize_vars(self):
        self.initialize_dist_vars()
        self.initialize_dist_list_vars()
        self.initialize_task_vars()
        self.initialize_traj_vars()

    def initialize_dist_vars(self):
        dist = self.policy.distribution

        self._old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }

        self._state_info_vars = {
            k: tf.placeholder(
                tf.float32, shape=[None] * (1 + 1) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }

        obs_flat = flatten_batch(self._obs_var, name="obs_flat")
        latent_flat = flatten_batch(self._latent_var, name="latent_flat")
        self._state_info_flat = flatten_batch_dict(
            self._state_info_vars, name="state_info_flat")

        self._dist_info_vars = self.policy.dist_info_sym(
            {
                self.policy.env_input_var: obs_flat,
                self.policy.latent_mean_var: latent_flat
            },
            self._state_info_flat,
            name="dist_info_vars")

    def initialize_dist_list_vars(self):
        dist = self.policy.distribution
        self._old_dist_info_vars_list = [
            self._old_dist_info_vars[k] for k in dist.dist_info_keys
        ]
        self._state_info_vars_list = [
            self._state_info_vars[k] for k in self.policy.state_info_keys
        ]

    def initialize_task_vars(self):
        self._task_enc_state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='task_enc_%s' % k)
            for k, shape in self.policy.embedding.state_info_specs
        }
        self._task_enc_state_info_vars_list = [
            self._task_enc_state_info_vars[k]
            for k in self.policy.embedding.state_info_keys
        ]

        self._task_enc_dist = self.policy.embedding.distribution

        self._task_enc_old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='task_enc_old_%s' % k)
            for k, shape in self._task_enc_dist.dist_info_specs
        }
        self._task_enc_old_dist_info_vars_list = [
            self._task_enc_old_dist_info_vars[k]
            for k in self._task_enc_dist.dist_info_keys
        ]

    def initialize_traj_vars(self):
        self._traj_enc_state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='traj_enc_%s' % k)
            for k, shape in self.traj_encoder.state_info_specs
        }
        self._traj_enc_state_info_vars_list = [
            self._traj_enc_state_info_vars[k]
            for k in self.traj_encoder.state_info_keys
        ]

        self._traj_enc_dist = self.traj_encoder.distribution

        self._traj_enc_old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='traj_enc_old_%s' % k)
            for k, shape in self._traj_enc_dist.dist_info_specs
        }
        self._traj_enc_old_dist_info_vars_list = [
            self._traj_enc_old_dist_info_vars[k]
            for k in self._traj_enc_dist.dist_info_keys
        ]

    ############################ input variables ##############################
    def build_policy_input(self):
        input_list = [
            self.policy.task_input_var,
            self.policy.env_input_var,
            self._obs_var,
            self._action_var,
            self._reward_var,
            self._baseline_var,
            self._trajectory_var,
            self._task_var,
            self._latent_var,
            self._valid_var,
        ] + self._state_info_vars_list \
          + self._old_dist_info_vars_list \
          + self._task_enc_state_info_vars_list \
          + self._task_enc_old_dist_info_vars_list \
          + self._traj_enc_state_info_vars_list \
          + self._traj_enc_old_dist_info_vars_list

        return input_list

    def build_traj_enc_input(self):
        input_list = [
            self._trajectory_var,
            self._latent_var,
            self._valid_var,
        ]

        return input_list

    ############################ loss #########################################
    def build_loss(self):
        task_enc_entropy, traj_ll, pol_entropy = self.get_entropy()

        rewards = self.get_rewards(task_enc_entropy, traj_ll, pol_entropy)

        surr_loss, pol_mean_kl = self.get_task_loss(task_enc_entropy, traj_ll,
                                                    pol_entropy, rewards)
        traj_enc_loss = self.get_traj_loss(task_enc_entropy, traj_ll,
                                           pol_entropy)
        return surr_loss, pol_mean_kl, traj_enc_loss, rewards

    def get_entropy(self):
        dist = self.policy.distribution
        traj_flat = flatten_batch(self._trajectory_var, name="traj_flat")
        latent_flat = flatten_batch(self._latent_var, name="latent_flat")

        # Calculate entropy terms
        # 1. Task encoder total entropy
        with tf.variable_scope('task_encoder_entropy'):
            task_dim = self.policy.embedding.input_space.flat_dim
            all_task_one_hots = tf.one_hot(
                np.arange(task_dim), task_dim, name="all_task_one_hots")
            all_task_dists = self.policy.embedding.dist_info_sym(
                all_task_one_hots, name="all_task_dists")
            all_task_entropies = self.policy.embedding.entropy_sym(
                all_task_dists, name="all_task_entropies")
            task_enc_entropy = tf.reduce_mean(
                all_task_entropies, name="task_enc_entropy")

        # 2. Trajectory encoder log-likelihoods (cross-entropies)
        with tf.variable_scope('traj_encoder_ce'):
            traj_ll_flat = self.traj_encoder.log_likelihood_sym(
                traj_flat, latent_flat, name="traj_ll_flat")
            traj_ll = tf.reshape(
                traj_ll_flat, [-1, self.max_path_length], name="traj_ll")

        # 3. Policy path entropies
        with tf.variable_scope('policy_entropy'):
            pol_entropy_flat = dist.entropy_sym(
                self._dist_info_vars, name="pol_entropy_flat")
            pol_entropy = tf.reshape(
                pol_entropy_flat, [-1, self.max_path_length],
                name="pol_entropy")

        self._all_task_entropies = all_task_entropies
        self._all_task_one_hots = all_task_one_hots
        self._all_task_dists = all_task_dists
        self._task_enc_entropy = task_enc_entropy
        self._traj_ll_flat = traj_ll_flat
        self._traj_ll = traj_ll
        self._pol_entropy_flat = pol_entropy_flat
        self._pol_entropy = pol_entropy

        return task_enc_entropy, traj_ll, pol_entropy

    def get_task_loss(self, task_enc_entropy, traj_ll, pol_entropy, rewards):

        dist = self.policy.distribution
        act_flat = flatten_batch(self._action_var, name="act_flat")
        valid_flat = flatten_batch(self._valid_var, name="valid_flat")
        old_dist_info_flat = flatten_batch_dict(
            self._old_dist_info_vars, name="old_dist_info_flat")

        with tf.variable_scope("policy_loss"):
            with tf.variable_scope("advantages"):
                # Prepare convolutional IIR filter to calculate advantages
                gamma_lambda = tf.constant(
                    float(self.discount) * float(self.gae_lambda),
                    dtype=tf.float32,
                    shape=[self.max_path_length, 1, 1])
                advantage_filter = tf.cumprod(gamma_lambda, exclusive=True)

                # Calculate deltas
                pad = tf.zeros_like(self._baseline_var[:, :1])
                baseline_shift = tf.concat([self._baseline_var[:, 1:], pad], 1)
                deltas = rewards + \
                         (self.discount * baseline_shift) - \
                         self._baseline_var
                # Convolve deltas with the discount filter to get advantages
                deltas_pad = tf.expand_dims(
                    tf.concat([deltas, tf.zeros_like(deltas[:, :-1])], axis=1),
                    axis=2)
                adv = tf.nn.conv1d(
                    deltas_pad, advantage_filter, stride=1, padding='VALID')
                advantages = tf.reshape(adv, [-1])
            adv_flat = flatten_batch(advantages, name="adv_flat")

            # Filter valid timesteps
            action_valid = filter_valids(
                act_flat, valid_flat, name="action_valid")
            state_info_valid = filter_valids_dict(
                self._state_info_vars, valid_flat, name="state_info_valid")
            old_dist_info_vars_valid = filter_valids_dict(
                old_dist_info_flat,
                valid_flat,
                name="old_dist_info_vars_valid")
            adv_valid = filter_valids(adv_flat, valid_flat, name="adv_valid")
            dist_info_vars_valid = filter_valids_dict(
                self._dist_info_vars, valid_flat, name="dist_info_vars_valid")

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                with tf.variable_scope("center_adv"):
                    mean, var = tf.nn.moments(adv_valid, axes=[0])
                    adv_valid = tf.nn.batch_normalization(
                        adv_valid, mean, var, 0, 1, eps)
            if self.positive_adv:
                with tf.variable_scope("positive_adv"):
                    m = tf.reduce_min(adv_valid)
                    adv_valid = (adv_valid - m) + eps

            # Calculate loss function and KL divergence
            with tf.variable_scope("kl"):
                kl = dist.kl_sym(old_dist_info_vars_valid,
                                 dist_info_vars_valid)
                pol_mean_kl = tf.reduce_mean(kl)

            with tf.variable_scope("surr_loss"):
                lr = dist.likelihood_ratio_sym(
                    action_valid,
                    old_dist_info_vars_valid,
                    dist_info_vars_valid,
                    name="lr")
                surr_loss = -tf.reduce_mean(lr * adv_valid) - \
                            (self.task_enc_ent_coeff * task_enc_entropy)

            self._dist_info_vars_valid = dist_info_vars_valid
            self._kl = kl
            self._lr = lr

        return surr_loss, pol_mean_kl

    def get_traj_loss(self, task_encoder_entropy, traj_ll, pol_entropy):
        traj_enc_dist = self.traj_encoder.distribution

        traj_flat = flatten_batch(self._trajectory_var, name="traj_flat")
        valid_flat = flatten_batch(self._valid_var, name="valid_flat")

        with tf.variable_scope("traj_enc_loss"):
            # Calculate loss
            traj_gammas = tf.constant(
                float(self.discount),
                dtype=tf.float32,
                shape=[self.max_path_length])
            traj_discounts = tf.cumprod(
                traj_gammas, exclusive=True, name="traj_discounts")
            discount_traj_ll = traj_discounts * traj_ll
            discount_traj_ll_flat = flatten_batch(
                discount_traj_ll, name="discount_traj_ll_flat")
            discount_traj_ll_valid = filter_valids(
                discount_traj_ll_flat,
                valid_flat,
                name="discount_traj_ll_valid")

            traj_enc_loss = -tf.reduce_mean(
                discount_traj_ll_valid, name="traj_enc_loss")

            # Flatten input variables
            traj_enc_state_info_flat = flatten_batch_dict(
                self._traj_enc_state_info_vars,
                name="traj_enc_state_info_flat")
            traj_enc_old_dist_info_flat = flatten_batch_dict(
                self._traj_enc_old_dist_info_vars,
                name="traj_enc_old_dist_info_flat")

            # Calculate task encoder distributions for each timestep
            traj_enc_dist_info_vars = self.traj_encoder.dist_info_sym(
                traj_flat,
                traj_enc_state_info_flat,
                name="traj_enc_dist_info_vars")

            # # Filter for valid time steps
            # traj_enc_old_dist_info_valid = filter_valids_dict(
            #     traj_enc_old_dist_info_flat,
            #     valid_flat,
            #     name="traj_enc_old_dist_info_valid")
            # traj_enc_dist_info_vars_valid = filter_valids_dict(
            #     traj_enc_dist_info_vars,
            #     valid_flat,
            #     name="traj_enc_dist_info_vars_valid")

            # # Calculate KL divergence

        return traj_enc_loss

    def get_rewards(self, task_encoder_entropy, traj_ll, pol_entropy):
        with tf.variable_scope("rewards"):
            # Augment the path rewards with entropy terms
            rewards = self._reward_var + \
                      (self.traj_enc_ent_coeff * traj_ll) + \
                      (self.policy_ent_coeff * pol_entropy)

        return rewards

    ############################ return variables #############################
    def build_returns(self, rewards):
        with tf.variable_scope("returns"):
            gamma = tf.constant(
                float(self.discount),
                dtype=tf.float32,
                shape=[self.max_path_length, 1, 1])
            return_filter = tf.cumprod(gamma, exclusive=True)
            rewards_pad = tf.expand_dims(
                tf.concat([rewards, tf.zeros_like(rewards[:, :-1])], axis=1),
                axis=2)
            returns = tf.nn.conv1d(
                rewards_pad, return_filter, stride=1, padding='VALID')
        return returns

    ############################ train ########################################
    def get_training_input(self, samples_data):
        tasks = np.reshape(samples_data["tasks"],
                           (-1, self.policy.task_space.flat_dim))
        obs = np.reshape(samples_data["observations"],
                         (-1, self.policy.observation_space.flat_dim))
        policy_input_values = (tasks, obs)
        policy_input_values += tuple(
            ext.extract(samples_data, 'observations', 'actions', 'rewards',
                        'baselines', 'trajectories', 'tasks', 'latents',
                        'valids'))
        # add policy params
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [
            agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]
        policy_input_values += tuple(state_info_list) + tuple(dist_info_list)
        # add task encoder params
        latent_infos = samples_data["latent_infos"]
        task_enc_state_info_list = [
            latent_infos[k] for k in self.policy.embedding.state_info_keys
        ]
        task_enc_dist_info_list = [
            latent_infos[k]
            for k in self.policy.embedding.distribution.dist_info_keys
        ]
        policy_input_values += tuple(task_enc_state_info_list) + tuple(
            task_enc_dist_info_list)
        # add trajectory encoder params
        trajectory_infos = samples_data["trajectory_infos"]
        traj_enc_state_info_list = [
            trajectory_infos[k] for k in self.traj_encoder.state_info_keys
        ]
        traj_enc_dist_info_list = [
            trajectory_infos[k]
            for k in self.traj_encoder.distribution.dist_info_keys
        ]
        policy_input_values += tuple(traj_enc_state_info_list) + tuple(
            traj_enc_dist_info_list)

        traj_enc_input_values = [
            samples_data['trajectories'], samples_data['latents'],
            samples_data['valids']
        ]

        return policy_input_values, traj_enc_input_values

    def get_feed(self, samples_data):
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [
            agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]

        # add trajectory encoder params
        trajectory_infos = samples_data["trajectory_infos"]
        traj_enc_state_info_list = [
            trajectory_infos[k] for k in self.traj_encoder.state_info_keys
        ]
        traj_enc_dist_info_list = [
            trajectory_infos[k]
            for k in self.traj_encoder.distribution.dist_info_keys
        ]

        feed = {
            self._obs_var: samples_data['observations'],
            self._task_var: samples_data['tasks'],
            self._action_var: samples_data['actions'],
            self._reward_var: samples_data['rewards'],
            self._baseline_var: samples_data['baselines'],
            self._trajectory_var: samples_data['trajectories'],
            self._latent_var: samples_data['latents'],
            self._valid_var: samples_data['valids'],
        }
        for idx, v in enumerate(self._state_info_vars_list):
            feed[v] = state_info_list[idx]
        for idx, v in enumerate(self._old_dist_info_vars_list):
            feed[v] = dist_info_list[idx]
        for idx, v in enumerate(self._traj_enc_state_info_vars_list):
            feed[v] = traj_enc_state_info_list[idx]
        for idx, v in enumerate(self._traj_enc_old_dist_info_vars_list):
            feed[v] = traj_enc_dist_info_list[idx]

        return feed

    def optimize(self, samples_data):
        policy_input_values, traj_enc_input_values = self.get_training_input(
            samples_data)

        samples_data = self.evaluate(policy_input_values, samples_data)

        self.train_task(policy_input_values)
        self.train_traj(traj_enc_input_values)

        self.visualize_distribution()

        return samples_data

    def train_task(self, all_input_values):
        # Joint optimization of policy and task encoder
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(all_input_values)
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(all_input_values)
        logger.log("Optimizing")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        logger.dump_tabular()

        return loss_after

    def train_traj(self, traj_enc_input):
        # Optimize trajectory encoder
        logger.log("Optimizing trajectory encoder...")

        traj_enc_loss_before = self.traj_enc_optimizer.loss(traj_enc_input)
        logger.record_tabular('TrajEncoder/Loss', traj_enc_loss_before)
        self.traj_enc_optimizer.optimize(traj_enc_input)
        traj_enc_loss_after = self.traj_enc_optimizer.loss(traj_enc_input)
        logger.record_tabular('TrajEncoder/dLoss',
                              traj_enc_loss_before - traj_enc_loss_after)
        logger.dump_tabular()

        return traj_enc_loss_after

    def evaluate(self, all_input_values, samples_data):
        # Everything else
        rewards_tensor = self.f_rewards(*all_input_values)
        returns_tensor = self.f_returns(*all_input_values)
        returns_tensor = np.squeeze(returns_tensor)  # TODO
        # TODO: check the squeeze/dimension handling for both convolutions

        paths = samples_data['paths']
        valids = samples_data['valids']
        baselines = [path['baselines'] for path in paths]
        env_rewards = [path['rewards'] for path in paths]
        env_rewards = tensor_utils.concat_tensor_list(env_rewards.copy())
        env_returns = [path['returns'] for path in paths]
        env_returns = tensor_utils.concat_tensor_list(env_returns.copy())

        # Recompute parts of samples_data
        aug_rewards = []
        aug_returns = []
        for rew, ret, val, path in zip(rewards_tensor, returns_tensor, valids,
                                       paths):
            path['rewards'] = rew[val.astype(np.bool)]
            path['returns'] = ret[val.astype(np.bool)]
            aug_rewards.append(path['rewards'])
            aug_returns.append(path['returns'])
        aug_rewards = tensor_utils.concat_tensor_list(aug_rewards)
        aug_returns = tensor_utils.concat_tensor_list(aug_returns)
        samples_data['rewards'] = aug_rewards
        samples_data['returns'] = aug_returns

        # Calculate effect of the entropy terms
        d_rewards = np.sqrt(np.sum((env_rewards - aug_rewards)**2))
        d_returns = np.sqrt(np.sum((env_returns - aug_returns)**2))
        logger.record_tabular('dAugmentedRewards', d_rewards)
        logger.record_tabular('dAugmentedReturns', d_returns)

        # Calculate explained variance
        ev = special.explained_variance_1d(
            np.concatenate(baselines), aug_returns)
        logger.record_tabular('ExplainedVariance', ev)

        # Fit baseline
        logger.log("Fitting baseline...")
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

        task_enc_rmse = (samples_data['trajectory_infos']['mean'] -
                         samples_data['latents'])**2.
        task_enc_rmse = np.sqrt(task_enc_rmse.mean())
        logger.record_tabular('TrajEncoder/RMSE', task_enc_rmse)

        #traj_enc_loss = self.train_traj(feed)
        task_enc_rmse = (samples_data['trajectory_infos']['mean'] -
                         samples_data['latents'])**2.
        task_enc_rmse = np.sqrt(task_enc_rmse.mean())
        logger.record_tabular('TrajEncoder/RMSE', task_enc_rmse)

        return samples_data

    # Visualize task embedding distributions
    def visualize_distribution(self):
        #TODO(@junchao) implement logger counterpart for distribution histograms
        if not hasattr(
                logger,
                '_tensorboard_writer') or logger._tensorboard_writer is None:
            return

        num_tasks = self.policy.task_space.flat_dim
        all_tasks = np.eye(num_tasks, num_tasks)
        for l in range(max(self.policy.latent_space.shape)):
            latent_distributions = []
            for t in range(num_tasks):
                _, latent_info = self.policy.get_latent(all_tasks[t, :])
                latent_distributions.append(
                    tf.random_normal(
                        shape=(1000, ),
                        mean=latent_info["mean"][l],
                        stddev=np.exp(latent_info["log_std"][l])))

                logger.record_tabular('TaskEncoder/Std/t=%i/%i' % (t, l),
                                      np.exp(latent_info["log_std"][l]))

            all_combined = tf.concat(latent_distributions, 0)
            if l == 0 and self.z_summary is None:
                self.z_summary = []
                self.z_summary_op = []
                for i in range(max(self.policy.latent_space.shape)):
                    self.z_summary.append(tf.Variable(all_combined))
                    self.z_summary_op.append(
                        tf.summary.histogram("TaskEmbedding/%i" % i,
                                             self.z_summary[-1]))

            x = tf.assign(self.z_summary[l], all_combined)
            tf.get_default_session().run(x)

        z_summary = tf.summary.merge(self.z_summary_op)
        z_summary = tf.get_default_session().run(z_summary)
        logger._tensorboard_writer.add_summary(
            z_summary, global_step=logger._tensorboard_default_step)
        logger._tensorboard_writer.flush()

    @overrides
    def optimize_policy(self, itr, **kwargs):
        paths = self.obtain_samples(itr)
        samples_data = self.process_samples(itr, paths)
        self.log_diagnostics(paths)

        samples_data = self.optimize(samples_data)
        # TODO: check the squeeze/dimension handling for both convolutions

        return self.get_itr_snapshot(itr, samples_data)

    def train(self, sess=None):
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())

        self.start_worker(sess)
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                params = self.optimize_policy(itr, )
                if self.plot and itr > self.plot_warmup_itrs:
                    rollout(
                        self.env,
                        self.policy,
                        animated=True,
                        max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
                logger.log("Saving snapshot...")
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('IterTime', time.time() - itr_start_time)
                logger.record_tabular('Time', time.time() - start_time)
        self.shutdown_worker()
        if created_session:
            sess.close()

    @overrides
    def get_itr_snapshot(self, itr, _samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            trajectory_encoder=self.traj_encoder,
        )
