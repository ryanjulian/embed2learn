import time

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import special
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger

from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.core.parameterized import JointParameterized

from sandbox.embed2learn.embeddings.base import Embedding
from sandbox.embed2learn.embeddings.utils import concat_spaces
from sandbox.embed2learn.samplers.task_embedding_sampler import TaskEmbeddingSampler
from sandbox.embed2learn.samplers.task_embedding_sampler import rollout
from sandbox.embed2learn.algos.utils import flatten_batch
from sandbox.embed2learn.algos.utils import flatten_batch_dict
from sandbox.embed2learn.algos.utils import filter_valids
from sandbox.embed2learn.algos.utils import filter_valids_dict


def _optimizer_or_default(optimizer, args):
    use_optimizer = optimizer
    use_args = args
    if use_optimizer is None:
        if use_args is None:
            use_args = dict(name="optimizer")
        use_optimizer = PenaltyLbfgsOptimizer(**use_args)
    return use_optimizer


class NPOTaskEmbedding(BatchPolopt, Serializable):
    """
    Natural Policy Optimization with Task Embeddings
    """

    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.01,
                 policy_ent_coeff=1e-2,
                 task_encoder=None,
                 task_encoder_ent_coeff=1e-5,
                 trajectory_encoder=None,
                 trajectory_encoder_ent_coeff=1e-3,
                 **kwargs):
        Serializable.quick_init(self, locals())
        assert kwargs['env'].task_space
        assert isinstance(task_encoder, Embedding)
        assert isinstance(trajectory_encoder, Embedding)

        self.task_encoder = task_encoder
        self.traj_encoder = trajectory_encoder

        # Joint optimizer for policy, task encoder and trajectory encoder
        self.optimizer = _optimizer_or_default(optimizer, optimizer_args)
        self.step_size = float(step_size)
        self.policy_ent_coeff = float(policy_ent_coeff)

        self.task_enc_ent_coeff = task_encoder_ent_coeff
        self.traj_enc_ent_coeff = trajectory_encoder_ent_coeff

        sampler_cls = TaskEmbeddingSampler
        sampler_args = dict(
            task_encoder=self.task_encoder,
            trajectory_encoder=self.traj_encoder,
        )
        super(NPOTaskEmbedding, self).__init__(
            sampler_cls=sampler_cls, sampler_args=sampler_args, **kwargs)

    @overrides
    def init_opt(self):
        loss, pol_mean_kl, task_enc_mean_kl, traj_enc_mean_kl, input_list = \
            self._build_opt()

        # Optimize policy, task_encoder and traj_encoder jointly
        targets = JointParameterized(
            components=[self.policy, self.task_encoder, self.traj_encoder])

        # TODO(): should we consider KL constraints for all three networks?
        self.optimizer.update_opt(
            loss=loss,
            target=targets,
            leq_constraint=(pol_mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl")

        return dict()

    def _build_opt(self):
        is_recurrent = int(self.policy.recurrent)
        if is_recurrent:
            raise NotImplementedError

        latent_obs_space = concat_spaces(self.task_encoder.latent_space,
                                         self.env.observation_space)

        #### Policy and loss function ##########################################

        # Input variables
        obs_var = latent_obs_space.new_tensor_variable(
            'obs',
            extra_dims=1 + 1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + 1,
        )
        reward_var = tensor_utils.new_tensor(
            'reward',
            ndim=1 + 1,
            dtype=tf.float32,
        )
        baseline_var = tensor_utils.new_tensor(
            'baseline',
            ndim=1 + 1,
            dtype=tf.float32,
        )
        trajectory_var = self.traj_encoder.input_space.new_tensor_variable(
            'trajectory',
            extra_dims=1 + 1,
        )
        latent_var = self.task_encoder.latent_space.new_tensor_variable(
            'latent',
            extra_dims=1 + 1,
        )
        valid_var = tf.placeholder(
            tf.float32, shape=[None, None], name="valid")

        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [
            old_dist_info_vars[k] for k in dist.dist_info_keys
        ]

        state_info_vars = {
            k: tf.placeholder(
                tf.float32, shape=[None] * (1 + 1) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [
            state_info_vars[k] for k in self.policy.state_info_keys
        ]

        # Flatten inputs and filter for valid timesteps
        # TODO: verify this works for tensors in general
        # Flatten
        obs_flat = flatten_batch(obs_var)
        act_flat = flatten_batch(action_var)
        traj_flat = flatten_batch(trajectory_var)
        latent_flat = flatten_batch(latent_var)
        valid_flat = flatten_batch(valid_var)
        state_info_flat = flatten_batch_dict(state_info_vars)
        old_dist_info_flat = flatten_batch_dict(old_dist_info_vars)

        # Calculate policy distributions for each timestep
        # TODO: may need to freeze this for all three op steps
        dist_info_vars = self.policy.dist_info_sym(obs_flat, state_info_flat)

        # Calculate entropy terms
        # 1. Task encoder total entropy
        task_dim = self.task_encoder.input_space.flat_dim
        all_task_one_hots = tf.one_hot(np.arange(task_dim), task_dim)
        all_task_dists = self.task_encoder.dist_info_sym(all_task_one_hots)
        all_task_entropies = self.task_encoder.entropy_sym(all_task_dists)
        task_enc_entropy = tf.reduce_mean(all_task_entropies)

        # 2. Trajectory encoder log likelihoods
        traj_ll_flat = self.traj_encoder.log_likelihood_sym(
            traj_flat, latent_flat)
        traj_ll = tf.reshape(traj_ll_flat, [-1, self.max_path_length])

        # 3. Policy encoder path entropies
        pol_entropy_flat = dist.entropy_sym(dist_info_vars)
        pol_entropy = tf.reshape(pol_entropy_flat, [-1, self.max_path_length])

        # Augment the path rewards with entropy terms
        rewards = reward_var + \
                  (self.traj_enc_ent_coeff * traj_ll) + \
                  (self.policy_ent_coeff * pol_entropy)

        # TODO(gh/17): this could be split into some symbolic ops and
        # contributed to tensor_utils
        #
        # Calculate advantages
        #
        # Advantages are a discounted cumulative sum.
        #
        # The discount cumulative sum can be represented as an IIR filter on the
        # reversed input vectors, i.e.
        #    y[t] - discount*y[t+1] = x[t]
        #        or
        #    rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
        #
        # Given the time-domain IIR filter step response, we can calculate the
        # filter response to our signal by convolving the signal with the filter
        # response function. The time-domain IIR step response is calculated
        # below as discount_filter:
        #     discount_filter := [1, discount, discount^2, ..., discount^N-1]
        #     where the epsiode length is N.
        #
        # We convolve discount_filter with the reversed time-domain signal
        # deltas to get calculate the reversed advantages:
        #     rev(advantages) = discount_filter (X) rev(deltas)
        #
        # TensorFlow's tf.nn.conv1d op is not a true convolution, but actually
        # a cross-correlation, so its input and output are already implicitly
        # reversed for us.
        #    advantages = discount_filter (tf.nn.conv1d) deltas
        #

        # Prepare convolutional IIR filter to calculate advantages
        gamma_lambda = tf.constant(
            float(self.discount) * float(self.gae_lambda),
            dtype=tf.float32,
            shape=[self.max_path_length, 1, 1])
        advantage_filter = tf.cumprod(gamma_lambda, exclusive=True)

        # Calculate deltas
        pad = tf.zeros_like(baseline_var[:, :1])
        baseline_shift = tf.concat([baseline_var[:, 1:], pad], 1)
        deltas = rewards + \
                 (self.discount * baseline_shift) - \
                 baseline_var
        # Convolve deltas with the discount filter to get advantages
        deltas_pad = tf.expand_dims(
            tf.concat([deltas, tf.zeros_like(deltas[:, :-1])], axis=1), axis=2)
        adv = tf.nn.conv1d(
            deltas_pad, advantage_filter, stride=1, padding='VALID')
        advantages = tf.reshape(adv, [-1])
        adv_flat = flatten_batch(advantages)

        # Filter valid timesteps
        obs_valid = filter_valids(obs_flat, valid_flat)
        act_valid = filter_valids(act_flat, valid_flat)
        state_info_valid = filter_valids_dict(state_info_vars, valid_flat)
        old_dist_info_valid = filter_valids_dict(old_dist_info_flat,
                                                 valid_flat)
        adv_valid = filter_valids(adv_flat, valid_flat)
        dist_info_vars_valid = filter_valids_dict(dist_info_vars, valid_flat)

        # Optionally normalize advantages
        eps = tf.constant(1e-8, dtype=tf.float32)
        if self.center_adv:
            mean, var = tf.nn.moments(adv_valid, axes=[0])
            adv_valid = tf.nn.batch_normalization(adv_valid, mean, var, 0, 1,
                                                  eps)

        if self.positive_adv:
            m = tf.reduce_min(adv_valid)
            adv_valid = (adv_valid - m) + eps

        # Calculate loss function and KL divergence
        kl = dist.kl_sym(old_dist_info_valid, dist_info_vars_valid)
        lr = dist.likelihood_ratio_sym(act_valid, old_dist_info_valid,
                                       dist_info_vars_valid)
        pol_mean_kl = tf.reduce_mean(kl)
        surr_loss = -tf.reduce_mean(lr * adv_valid) - \
                    (self.task_enc_ent_coeff * task_enc_entropy)

        #### Returns (for the baseline) ########################################
        # This uses the same filtering trick as above to calculate the
        # discounted cumulative sum
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

        #### Task encoder KL divergence ########################################
        # Input variables
        task_var = self.task_encoder.input_space.new_tensor_variable(
            'task',
            extra_dims=1 + 1,
        )

        task_enc_state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='task_enc_%s' % k)
            for k, shape in self.task_encoder.state_info_specs
        }
        task_enc_state_info_vars_list = [
            task_enc_state_info_vars[k]
            for k in self.task_encoder.state_info_keys
        ]

        task_enc_dist = self.task_encoder.distribution

        task_enc_old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='task_enc_old_%s' % k)
            for k, shape in task_enc_dist.dist_info_specs
        }
        task_enc_old_dist_info_vars_list = [
            task_enc_old_dist_info_vars[k]
            for k in task_enc_dist.dist_info_keys
        ]

        # Flatten input variables
        task_flat = flatten_batch(task_var)
        task_enc_state_info_flat = flatten_batch_dict(task_enc_state_info_vars)
        task_enc_old_dist_info_flat = flatten_batch_dict(
            task_enc_old_dist_info_vars)

        # Calculate task encoder distributions for each timestep
        task_enc_dist_info_vars = self.task_encoder.dist_info_sym(
            task_flat, task_enc_state_info_flat)

        # Filter for valid time steps
        task_enc_old_dist_info_valid = filter_valids_dict(
            task_enc_old_dist_info_flat, valid_flat)
        task_enc_dist_info_vars_valid = filter_valids_dict(
            task_enc_dist_info_vars, valid_flat)

        # Calculate KL divergence
        task_enc_kl = task_enc_dist.kl_sym(task_enc_old_dist_info_valid,
                                           task_enc_dist_info_vars_valid)
        task_enc_mean_kl = tf.reduce_mean(task_enc_kl)

        #### Trajectory encoder KL divergence ##################################
        traj_enc_state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='traj_enc_%s' % k)
            for k, shape in self.traj_encoder.state_info_specs
        }
        traj_enc_state_info_vars_list = [
            traj_enc_state_info_vars[k]
            for k in self.traj_encoder.state_info_keys
        ]

        traj_enc_dist = self.traj_encoder.distribution

        traj_enc_old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='traj_enc_old_%s' % k)
            for k, shape in traj_enc_dist.dist_info_specs
        }
        traj_enc_old_dist_info_vars_list = [
            traj_enc_old_dist_info_vars[k]
            for k in traj_enc_dist.dist_info_keys
        ]

        # Flatten input variables
        traj_enc_state_info_flat = flatten_batch_dict(traj_enc_state_info_vars)
        traj_enc_old_dist_info_flat = flatten_batch_dict(
            traj_enc_old_dist_info_vars)

        # Calculate task encoder distributions for each timestep
        traj_enc_dist_info_vars = self.traj_encoder.dist_info_sym(
            traj_flat, traj_enc_state_info_flat)

        # Filter for valid time steps
        traj_enc_old_dist_info_valid = filter_valids_dict(
            traj_enc_old_dist_info_flat, valid_flat)
        traj_enc_dist_info_vars_valid = filter_valids_dict(
            traj_enc_dist_info_vars, valid_flat)

        # Calculate KL divergence
        traj_enc_kl = traj_enc_dist.kl_sym(traj_enc_old_dist_info_valid,
                                           traj_enc_dist_info_vars_valid)
        traj_enc_mean_kl = tf.reduce_mean(traj_enc_kl)

        #### Input list ########################################################

        input_list = [
            obs_var,
            action_var,
            reward_var,
            baseline_var,
            trajectory_var,
            task_var,
            latent_var,
            valid_var,
        ] + state_info_vars_list + old_dist_info_vars_list \
          + task_enc_state_info_vars_list + task_enc_old_dist_info_vars_list \
          + traj_enc_state_info_vars_list + traj_enc_old_dist_info_vars_list

        #### DEBUG #############################################################
        # Inputs
        self._obs_var = obs_var
        self._action_var = action_var
        self._reward_var = reward_var
        self._baseline_var = baseline_var
        self._trajectory_var = trajectory_var
        self._task_var = task_var
        self._latent_var = latent_var
        self._valid_var = valid_var
        self._state_info_vars_list = state_info_vars_list
        self._old_dist_info_vars_list = old_dist_info_vars_list
        self._task_enc_state_info_vars_list = task_enc_state_info_vars_list
        self._task_enc_old_dist_info_vars_list = task_enc_old_dist_info_vars_list
        self._traj_enc_state_info_vars_list = traj_enc_state_info_vars_list
        self._traj_enc_old_dist_info_vars_list = traj_enc_old_dist_info_vars_list
        # Outputs
        deltas_flat = flatten_batch(deltas)
        deltas_flat = filter_valids(deltas_flat, valid_flat)
        self._f_adv = adv_valid
        self._f_deltas = deltas_flat
        self._f_base_shift = baseline_shift
        self._dist_info_vars = dist_info_vars
        self._dist_info_vars_valid = dist_info_vars_valid
        self._all_task_one_hots = all_task_one_hots
        self._all_task_dists = all_task_dists
        self._all_task_entropies = all_task_entropies
        self._task_enc_entropy = task_enc_entropy
        self._pol_entropy_flat = pol_entropy_flat
        self._pol_entropy = pol_entropy
        self._traj_ll_flat = traj_ll_flat
        self._traj_ll = traj_ll
        self._kl = kl
        self._lr = lr
        self._pol_mean_kl = pol_mean_kl
        self._surr_loss = surr_loss
        self._task_enc_mean_kl = task_enc_mean_kl
        self._traj_enc_mean_kl = traj_enc_mean_kl

        # DEBUG CPU VERSION ####################################################
        cpu_obs_var = latent_obs_space.new_tensor_variable(
            'obs_cpu',
            extra_dims=1 + is_recurrent,
        )
        cpu_action_var = self.env.action_space.new_tensor_variable(
            'action_cpu',
            extra_dims=1 + is_recurrent,
        )
        cpu_advantage_var = tensor_utils.new_tensor(
            'advantage_cpu',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        cpu_old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='cpu_old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        cpu_old_dist_info_vars_list = [
            cpu_old_dist_info_vars[k] for k in dist.dist_info_keys
        ]

        cpu_state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='cpu_old_state_%s' % k)
            for k, shape in self.policy.state_info_specs
        }
        cpu_state_info_vars_list = [
            cpu_state_info_vars[k] for k in self.policy.state_info_keys
        ]

        cpu_dist_info_vars = self.policy.dist_info_sym(cpu_obs_var,
                                                       cpu_state_info_vars)
        cpu_kl = dist.kl_sym(cpu_old_dist_info_vars, cpu_dist_info_vars)
        cpu_lr = dist.likelihood_ratio_sym(
            cpu_action_var, cpu_old_dist_info_vars, cpu_dist_info_vars)
        cpu_mean_kl = tf.reduce_mean(cpu_kl)
        cpu_surr_loss = -tf.reduce_mean(cpu_lr * cpu_advantage_var)

        # Inputs
        self._cpu_obs_var = cpu_obs_var
        self._cpu_action_var = cpu_action_var
        self._cpu_advantage_var = cpu_advantage_var
        self._cpu_state_info_vars_list = cpu_state_info_vars_list
        self._cpu_old_dist_info_vars_list = cpu_old_dist_info_vars_list

        # Outputs
        self._cpu_dist_info_vars = cpu_dist_info_vars
        self._cpu_kl = cpu_kl
        self._cpu_lr = cpu_lr
        self._cpu_mean_kl = cpu_mean_kl
        self._cpu_surr_loss = cpu_surr_loss
        #######################################################################

        self.f_rewards = tensor_utils.compile_function(
            input_list, rewards, log_name="f_rewards")
        self.f_returns = tensor_utils.compile_function(
            input_list, returns, log_name="f_returns")

        return surr_loss, pol_mean_kl, task_enc_mean_kl, traj_enc_mean_kl, \
               input_list

    @overrides
    def optimize_policy(self, itr, samples_data):

        # Collect input values
        all_input_values = tuple(
            ext.extract(samples_data, 'observations', 'actions', 'rewards',
                        'baselines', 'trajectories', 'tasks', 'latents',
                        'valids'))
        # add policy params
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [
            agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        # add task encoder params
        latent_infos = samples_data["latent_infos"]
        task_enc_state_info_list = [
            latent_infos[k] for k in self.task_encoder.state_info_keys
        ]
        task_enc_dist_info_list = [
            latent_infos[k]
            for k in self.task_encoder.distribution.dist_info_keys
        ]
        all_input_values += tuple(task_enc_state_info_list) + tuple(
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
        all_input_values += tuple(traj_enc_state_info_list) + tuple(
            traj_enc_dist_info_list)

        #### DEBUG #############################################################
        # for k, v in samples_data.items():
        #     if hasattr(v, 'shape'):
        #         print('{}: {}'.format(k,v.shape))
        #     if isinstance(v, dict):
        #         for l, w in v.items():
        #             print('{}/{}: {}'.format(k, l, w.shape))

        # all_input_values = tuple(
        #     ext.extract(samples_data, "observations", "actions", "advantages"))

        # calculate cpu values
        np.set_printoptions(threshold=np.inf)
        cpu_agent_infos = samples_data["cpu_agent_infos"]
        cpu_state_info_list = [
            cpu_agent_infos[k] for k in self.policy.state_info_keys
        ]
        cpu_dist_info_list = [
            cpu_agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]
        feed = {
            self._obs_var: samples_data['observations'],
            self._action_var: samples_data['actions'],
            self._reward_var: samples_data['rewards'],
            self._baseline_var: samples_data['baselines'],
            self._trajectory_var: samples_data['trajectories'],
            self._task_var: samples_data['tasks'],
            self._latent_var: samples_data['latents'],
            self._valid_var: samples_data['valids'],
            self._cpu_obs_var: samples_data['cpu_obs'],
            self._cpu_action_var: samples_data['cpu_act'],
            self._cpu_advantage_var: samples_data['cpu_adv'],
        }
        for idx, v in enumerate(self._state_info_vars_list):
            feed[v] = state_info_list[idx]
        for idx, v in enumerate(self._old_dist_info_vars_list):
            feed[v] = dist_info_list[idx]
        for idx, v in enumerate(self._cpu_state_info_vars_list):
            feed[v] = cpu_state_info_list[idx]
        for idx, v in enumerate(self._cpu_old_dist_info_vars_list):
            feed[v] = cpu_dist_info_list[idx]
        for idx, v in enumerate(self._task_enc_state_info_vars_list):
            feed[v] = task_enc_state_info_list[idx]
        for idx, v in enumerate(self._task_enc_old_dist_info_vars_list):
            feed[v] = task_enc_dist_info_list[idx]
        for idx, v in enumerate(self._traj_enc_state_info_vars_list):
            feed[v] = traj_enc_state_info_list[idx]
        for idx, v in enumerate(self._traj_enc_old_dist_info_vars_list):
            feed[v] = traj_enc_dist_info_list[idx]

        # for k, v in feed.items():
        #     if hasattr(v, 'shape'):
        #         print('{}: {}'.format(k,v.shape))
        #     elif isinstance(v, dict):
        #         for l, w in v.items():
        #             print('{}/{}: {}'.format(k, l, w.shape))
        #     else:
        #         print('Cannot find shape of {}'.format(k))

        # measure KL divergence between task 1 and 2, 3:
        task1 = np.zeros((3,), dtype=np.float32)
        task2 = np.zeros((3,), dtype=np.float32)
        task3 = np.zeros((3,), dtype=np.float32)
        task1[0] = 1
        task2[1] = 1
        task3[2] = 1
        _, latent_info1 = self.task_encoder.get_latent(task1)
        _, latent_info2 = self.task_encoder.get_latent(task2)
        _, latent_info3 = self.task_encoder.get_latent(task3)
        latent_info1 = flatten_batch_dict(latent_info1)
        latent_info2 = flatten_batch_dict(latent_info2)
        latent_info3 = flatten_batch_dict(latent_info3)
        dist = self.policy.distribution
        kl12 = dist.kl_sym(latent_info1, latent_info2)
        kl13 = dist.kl_sym(latent_info1, latent_info3)

        sess = tf.get_default_session()
        # Everything else
        gpu_steps = {
            'dist_info_vars': self._dist_info_vars,
            'dist_info_vars_valid': self._dist_info_vars_valid,
            'all_task_one_hots': self._all_task_one_hots,
            'all_task_dists': self._all_task_dists,
            'all_task_entropies': self._all_task_entropies,
            'task_enc_entropy': self._task_enc_entropy,
            'pol_entropy_flat': self._pol_entropy_flat,
            'pol_entropy': self._pol_entropy,
            'traj_ll_flat': self._traj_ll_flat,
            'traj_ll': self._traj_ll,
            'kl': self._kl,
            'lr': self._lr,
            'pol_mean_kl': self._pol_mean_kl,
            'task_enc_mean_kl': self._task_enc_mean_kl,
            'traj_enc_mean_kl': self._traj_enc_mean_kl,
            'surr_loss': self._surr_loss,
            'kl12': kl12,
            'kl13': kl13,
        }
        cpu_steps = {
            'dist_info_vars': self._cpu_dist_info_vars,
            'kl': self._cpu_kl,
            'lr': self._cpu_lr,
            'mean_kl': self._cpu_mean_kl,
            'surr_loss': self._cpu_surr_loss,
        }
        f_gpu, f_cpu = sess.run((gpu_steps, cpu_steps), feed_dict=feed)

        # Advantage step
        adv_tf = sess.run(self._f_adv, feed_dict=feed)
        adv_cpu = samples_data['cpu_adv']
        dadv = np.sqrt(np.sum((adv_cpu - adv_tf)**2))
        #print('adv_tf: {}'.format(adv_tf))
        #print('adv_cpu: {}'.format(adv_cpu))
        logger.record_tabular('dAdv', dadv)
        #print('mean(adv_cpu): {}'.format(np.mean(adv_cpu)))
        #print('mean(adv_tf): {}'.format(np.mean(adv_tf)))
        #print('std(adv_cpu): {}'.format(np.std(adv_cpu)))
        #print('std(adv_tf): {}'.format(np.std(adv_tf)))

        logger.record_tabular("KL task 1-2", f_gpu['kl12'])
        logger.record_tabular("KL task 1-3", f_gpu['kl13'])

        # policy entropy
        #print('dist_info_vars[log_std]: {}'.format(f_gpu['dist_info_vars']['log_std']))
        #print('pol_entropy: {}'.format(f_gpu['pol_entropy']))
        #print('rewards.shape: {}'.format(samples_data['rewards'].shape))
        #print('pol_entropy.shape: {}'.format(f_gpu['pol_entropy'].shape))

        # task encoder entropy
        # print('all_task_one_hots: {}'.format(f_gpu['all_task_one_hots']))
        # print('all_task_dists: {}'.format(f_gpu['all_task_dists']))
        # print('all_task_entropies: {}'.format(f_gpu['all_task_entropies']))
        # print('task_enc_entropy: {}'.format(f_gpu['task_enc_entropy']))

        # traj log likelihood
        #print('traj_ll_flat: {}'.format(f_gpu['traj_ll_flat']))
        #print('traj_ll: {}'.format(f_gpu['traj_ll']))
        #print('traj_ll.shape: {}'.format(f_gpu['traj_ll'].shape))

        # LR
        # dlr = np.sqrt(np.sum((f_cpu['lr'] - f_gpu['lr'])**2))
        # print('dLR: {}'.format(dlr))

        # surr_loss
        # print('CPU surr_loss: {}'.format(f_cpu['surr_loss']))
        # print('GPU surr_loss: {}'.format(f_gpu['surr_loss']))
        dsurr_loss = f_cpu['surr_loss'] - f_gpu['surr_loss']
        logger.record_tabular('dSurr_loss', dsurr_loss)

        # mean_kl
        # print('CPU mean_kl: {}'.format(f_cpu['mean_kl']))
        # print('GPU mean_kl: {}'.format(f_gpu['mean_kl']))

        # dist_info_vars
        # print('dist_info')
        # for k, v in f_cpu['dist_info_vars'].items():
        #     dVal = np.sqrt(np.sum((v - f_gpu['dist_info_vars_valid'][k])**2))
        #     print('d{}: {}'.format(k, dVal))

        # KL
        # print('CPU KL: {}'.format(f_cpu['kl']))
        # print('GPU KL: {}'.format(f_gpu['kl']))

        # Delta step
        delta_tf = sess.run(self._f_deltas, feed_dict=feed)
        delta_cpu = samples_data['cpu_deltas']
        ddelta = np.sqrt(np.sum((delta_cpu - delta_tf)**2))
        #print('deltas_tf: {}'.format(delta_tf))
        #print('deltas_cpu: {}'.format(delta_cpu))
        logger.record_tabular('dDelta', ddelta)

        # Baselines shift
        # base_shift = sess.run(self._f_base_shift, feed_dict=feed)
        # print('baselines.shape: {}:'.format(samples_data['baselines'].shape))
        #print('baselines_shift: {}:'.format(base_shift))

        ########################################################################

        # Baseline optimization ################################################
        # Get rewards and returns from TF
        # IMPORTANT: this must be calculated *before* any optimization, because
        # the values depend on the network parameters
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

        # Joint optimization of policy, task encoder, and trajectory encoder ###
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

        return dict()

    def train(self, sess=None):
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        sess.run(tf.global_variables_initializer())
        self.start_worker()
        start_time = time.time()
        for itr in range(self.start_itr, self.n_itr):
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                logger.log("Obtaining samples...")
                paths = self.obtain_samples(itr)
                logger.log("Processing samples...")
                samples_data = self.process_samples(itr, paths)
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                logger.log("Optimizing policy and embedding...")
                self.optimize_policy(itr, samples_data)
                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr,
                                               samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)

                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    rollout(
                        self.env,
                        self.policy,
                        self.task_encoder,
                        animated=True,
                        max_path_length=self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            task_encoder=self.task_encoder,
            trajectory_encoder=self.traj_encoder,
        )
