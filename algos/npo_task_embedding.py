import time

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import special
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.embed2learn.embeddings.gaussian_mlp_multitask_policy import GaussianMLPMultitaskPolicy
from sandbox.embed2learn.embeddings.multitask_policy import StochasticMultitaskPolicy

from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.core.parameterized import JointParameterized

from sandbox.embed2learn.embeddings.base import StochasticEmbedding
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
            use_args = dict()
        use_optimizer = ConjugateGradientOptimizer(**use_args)
    return use_optimizer


class NPOTaskEmbedding(BatchPolopt, Serializable):
    """
    Natural Policy Optimization with Task Embeddings
    """

    def __init__(self,
                 plot_warmup_itrs=0,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.01,
                 policy_ent_coeff=1e-2,
                 policy: GaussianMLPMultitaskPolicy = None,
                 task_encoder_ent_coeff=1e-5,
                 trajectory_encoder: StochasticEmbedding = None,
                 trajectory_encoder_optimizer=None,
                 trajectory_encoder_optimizer_args=None,
                 trajectory_encoder_ent_coeff=1e-3,
                 trajectory_encoder_step_size=0.01,
                 **kwargs):
        Serializable.quick_init(self, locals())
        assert kwargs['env'].task_space
        assert isinstance(policy, GaussianMLPMultitaskPolicy
                          )  # TODO change to StochasticMultitaskPolicy
        assert isinstance(trajectory_encoder, StochasticEmbedding)

        self.plot_warmup_itrs = plot_warmup_itrs
        
        # Optimizer for policy + task encoder
        self.optimizer = _optimizer_or_default(optimizer, optimizer_args)
        self.step_size = float(step_size)
        self.policy_ent_coeff = float(policy_ent_coeff)
        self.task_enc_ent_coeff = float(task_encoder_ent_coeff)

        self.traj_encoder = trajectory_encoder
        self.traj_enc_ent_coeff = trajectory_encoder_ent_coeff
        self.traj_enc_optimizer = _optimizer_or_default(
            trajectory_encoder_optimizer, trajectory_encoder_optimizer_args)
        self.traj_enc_step_size = float(trajectory_encoder_step_size)

        sampler_cls = TaskEmbeddingSampler
        sampler_args = dict(trajectory_encoder=self.traj_encoder, )
        super(NPOTaskEmbedding, self).__init__(
            sampler_cls=sampler_cls,
            sampler_args=sampler_args,
            policy=policy,
            **kwargs)
        self.summary_writer = self.sampler.summary_writer

    @overrides
    def init_opt(self):
        loss, pol_mean_kl, traj_enc_loss, traj_enc_mean_kl, input_list = self._build_opt()

        
        # Optimize policy and task encoder jointly
        # TODO check if this needs self.policy._embedding (it shouldn't)
        pol_embed = JointParameterized(components=[self.policy, self.policy._embedding])
        self.optimizer.update_opt(
            loss=loss,
            target=pol_embed,
            leq_constraint=(pol_mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl")

        # Optimize trajectory encoder separately
        self.traj_enc_optimizer.update_opt(
            loss=traj_enc_loss,
            target=self.traj_encoder,
            leq_constraint=(traj_enc_mean_kl, self.traj_enc_step_size),
            inputs=input_list,
            constraint_name="mean_kl",
        )


        return dict()

    def _build_opt(self):
        tf.variable_scope('npo_task_embedding').__enter__()

        is_recurrent = int(self.policy.recurrent)
        if is_recurrent:
            raise NotImplementedError

        #### Policy and loss function ##########################################
        # Input variables
        self._obs_var = self.policy.observation_space.new_tensor_variable( 'obs', extra_dims=1 + 1,)
        self._task_var = self.policy.task_space.new_tensor_variable( 'task', extra_dims=1 + 1,)
        self._action_var = self.env.action_space.new_tensor_variable( 'action', extra_dims=1 + 1,)
        self._reward_var = tensor_utils.new_tensor( 'reward', ndim=1 + 1, dtype=tf.float32,)
        self._baseline_var = tensor_utils.new_tensor( 'baseline', ndim=1 + 1, dtype=tf.float32,)
        self._trajectory_var = self.traj_encoder.input_space.new_tensor_variable( 'trajectory', extra_dims=1 + 1,)
        self._valid_var = tf.placeholder( tf.float32, shape=[None, None], name="valid") 


        self.initialize_vars()

        input_list = self.build_input()        
        surr_loss, pol_mean_kl, traj_enc_loss, traj_mean_kl, rewards = self.build_loss()
        returns = self.build_returns(rewards)


        # Outputs
        self._traj_mean_kl = traj_mean_kl
        self._traj_loss = traj_enc_loss
        self._pol_mean_kl = pol_mean_kl
        self._surr_loss = surr_loss

        # Functions
        self.f_rewards = tensor_utils.compile_function(
            input_list, rewards, log_name="f_rewards")
        self.f_returns = tensor_utils.compile_function(
            input_list, returns, log_name="f_returns")
        self.f_task_entropies = tensor_utils.compile_function(
            input_list, self._all_task_entropies, log_name="f_task_entropies")
        self.f_policy_entropy = tensor_utils.compile_function(
            input_list,
            tf.reduce_sum(self._pol_entropy * self._valid_var),
            log_name="f_policy_entropy")
        self.f_traj_cross_entropy = tensor_utils.compile_function(
            input_list,
            tf.reduce_sum(self._traj_ll * self._valid_var),
            log_name="f_traj_cross_entropy")


        return self._surr_loss, self._pol_mean_kl, self._traj_loss, self._traj_mean_kl, input_list  

    ############################ initialize base variables #################################
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

        with tf.variable_scope('flatten'):
            obs_flat = flatten_batch(self._obs_var)
            latent_flat = self.policy.latent_mean_var
            self._state_info_flat = flatten_batch_dict(self._state_info_vars)

        self._dist_info_vars = self.policy.dist_info_sym({
            self.policy.env_input_var: obs_flat,
            self.policy.latent_mean_var: latent_flat
        }, self._state_info_flat)


    def initialize_dist_list_vars(self):
        dist = self.policy.distribution        
        self._old_dist_info_vars_list = [ self._old_dist_info_vars[k] for k in dist.dist_info_keys ]
        self._state_info_vars_list = [ self._state_info_vars[k] for k in self.policy.state_info_keys ]



    def initialize_task_vars(self):
        self._task_enc_state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + 1) + list(shape),
                name='task_enc_%s' % k)
            for k, shape in self.policy.embedding.state_info_specs
        }
        self._task_enc_state_info_vars_list = [
            task_enc_state_info_vars[k]
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

    ############################ input variables ####################################################
    def build_input(self):
        input_list = [
            self.policy.task_input_var,
            self.policy.env_input_var,
            self._obs_var,
            self._action_var,
            self._reward_var,
            self._baseline_var,
            self._trajectory_var,
            self._task_var,
            self._valid_var,
        ] + self._state_info_vars_list + self._old_dist_info_vars_list \
          + self._task_enc_state_info_vars_list + self._task_enc_old_dist_info_vars_list \
          + self._traj_enc_state_info_vars_list + self._traj_enc_old_dist_info_vars_list

        return input_list

    ############################ loss ####################################################
    def build_loss(self):
        task_enc_entropy, traj_ll, pol_entropy = self.get_entropy()

        rewards = self.get_rewards(task_enc_entropy, traj_ll, pol_entropy)

        surr_loss, pol_mean_kl = self.get_task_loss(task_enc_entropy, traj_ll, pol_entropy, rewards)
        traj_enc_loss, traj_mean_kl = self.get_traj_loss(task_enc_entropy, traj_ll, pol_entropy)
        return surr_loss, pol_mean_kl, traj_enc_loss, traj_mean_kl, rewards

    def get_entropy(self):
        dist = self.policy.distribution        
        with tf.variable_scope('flatten'):
            traj_flat = flatten_batch(self._trajectory_var)
            latent_flat = self.policy.latent_mean_var

        with tf.variable_scope('entropies'):
            # Calculate entropy terms
            # 1. Task encoder total entropy
            with tf.variable_scope('task_encoder_entropy'):
                task_dim = self.policy.embedding.input_space.flat_dim
                all_task_one_hots = tf.one_hot(np.arange(task_dim), task_dim)
                all_task_dists = self.policy.embedding.dist_info_sym(
                    all_task_one_hots)
                all_task_entropies = self.policy.embedding.entropy_sym(
                    all_task_dists)
                task_enc_entropy = tf.reduce_mean(all_task_entropies)

            # 2. Trajectory encoder log-likelihoods (cross-entropies)
            with tf.variable_scope('traj_encoder_ce'):
                traj_ll_flat = self.traj_encoder.log_likelihood_sym(
                    traj_flat, latent_flat)
                traj_ll = tf.reshape(traj_ll_flat, [-1, self.max_path_length])

            # 3. Policy path entropies
            with tf.variable_scope('policy_entropy'):
                pol_entropy_flat = dist.entropy_sym(self._dist_info_vars)
                pol_entropy = tf.reshape(pol_entropy_flat,
                                         [-1, self.max_path_length])

        self._all_task_entropies = all_task_entropies
        self._all_task_one_hots = all_task_one_hots
        self._all_task_dists = all_task_dists
        self._task_enc_entropy = task_enc_entropy
        self._traj_ll_flat = traj_ll_flat
        self._traj_ll = traj_ll
        self._pol_entropy_flat = pol_entropy_flat
        self._pol_entropy = pol_entropy

        return task_enc_entropy, traj_ll,  pol_entropy

    def get_task_loss(self, task_enc_entropy, traj_ll, pol_entropy, rewards):
        
        dist = self.policy.distribution        
        with tf.variable_scope('flatten'):
            act_flat = flatten_batch(self._action_var)
            valid_flat = flatten_batch(self._valid_var)
            old_dist_info_flat = flatten_batch_dict(self._old_dist_info_vars)

        with tf.variable_scope('task_enc'):
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
            adv_flat = flatten_batch(advantages)

            # Filter valid timesteps
            action_valid = filter_valids(act_flat, valid_flat)
            state_info_valid = filter_valids_dict(self._state_info_vars, valid_flat)
            old_dist_info_vars_valid = filter_valids_dict(old_dist_info_flat,
                                                     valid_flat)
            adv_valid = filter_valids(adv_flat, valid_flat)
            dist_info_vars_valid = filter_valids_dict(self._dist_info_vars,
                                                      valid_flat)

            # Optionally normalize advantages
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                mean, var = tf.nn.moments(adv_valid, axes=[0])
                adv_valid = tf.nn.batch_normalization(adv_valid, mean, var, 0,
                                                      1, eps)
            if self.positive_adv:
                m = tf.reduce_min(adv_valid)
                adv_valid = (adv_valid - m) + eps
            
            # Calculate loss function and KL divergence
            kl = dist.kl_sym(old_dist_info_vars_valid, dist_info_vars_valid)
            lr = dist.likelihood_ratio_sym(action_valid, old_dist_info_vars_valid,
                                           dist_info_vars_valid)
            pol_mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr * adv_valid) - \
                        (self.task_enc_ent_coeff * task_enc_entropy)

            self._dist_info_vars_valid = dist_info_vars_valid
            self._kl = kl
            self._lr = lr

        return surr_loss, pol_mean_kl
    
    def get_traj_loss(self, task_encoder_entropy, traj_ll, pol_entropy):
        traj_enc_dist = self.traj_encoder.distribution
        with tf.variable_scope('flatten'):
            traj_flat = flatten_batch(self._trajectory_var)
            valid_flat = flatten_batch(self._valid_var)
        with tf.variable_scope('traj_enc'):
            # Calculate loss
            traj_gammas = tf.constant(
                float(self.discount),
                dtype=tf.float32,
                shape=[self.max_path_length])
            traj_discounts = tf.cumprod(traj_gammas, exclusive=True)
            discount_traj_ll = traj_discounts * traj_ll
            discount_traj_ll_flat = flatten_batch(discount_traj_ll)
            discount_traj_ll_valid = filter_valids(discount_traj_ll_flat,
                                                   valid_flat)

            traj_enc_loss = -tf.reduce_mean(discount_traj_ll_valid)

            # Flatten input variables
            traj_enc_state_info_flat = flatten_batch_dict(
                self._traj_enc_state_info_vars)
            traj_enc_old_dist_info_flat = flatten_batch_dict(
                self._traj_enc_old_dist_info_vars)

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
    
            self._traj_enc_mean_kl = traj_enc_mean_kl
    
        return traj_enc_loss, traj_enc_mean_kl

    def get_rewards(self, task_encoder_entropy, traj_ll, pol_entropy):
        with tf.variable_scope('advantages'):
            # Augment the path rewards with entropy terms
            rewards = self._reward_var + \
                      (self.traj_enc_ent_coeff * traj_ll) + \
                      (self.policy_ent_coeff * pol_entropy)

        return rewards


    ############################ return variables #########################################
    def build_returns(self, rewards):
        with tf.variable_scope('returns'):
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


    ############################ train ####################################################
    def get_training_input(self, samples_data):
        # Collect input values
        tasks = np.reshape(samples_data["tasks"],
                           (-1, self.policy.task_space.flat_dim))
        obs = np.reshape(samples_data["observations"],
                         (-1, self.policy.observation_space.flat_dim))
        all_input_values = (tasks, obs)
        all_input_values += tuple(
            ext.extract(
                samples_data,
                'observations',
                'actions',
                'rewards',
                'baselines',
                'trajectories',
                'tasks',
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
            latent_infos[k] for k in self.policy.embedding.state_info_keys
        ]
        task_enc_dist_info_list = [
            latent_infos[k]
            for k in self.policy.embedding.distribution.dist_info_keys
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

        # calculate cpu values
        np.set_printoptions(threshold=np.inf)
        return all_input_values, tasks, obs, dist_info_list, traj_enc_dist_info_list

    def optimize(self, samples_data):
        sess = tf.get_default_session()
        all_input_values, tasks, obs, dist_info_list, traj_enc_dist_info_list = self.get_training_input(samples_data)
        
        summary = tf.Summary()

        task_ents = self.f_task_entropies(*all_input_values)
        for i, v in enumerate(task_ents):
            logger.record_tabular('TaskEncoder/Entropy/t={}'.format(i), v)
        logger.record_tabular('TaskEncoder/Entropy', np.mean(task_ents))
        summary.value.add(tag='train/entropy_mean', simple_value=float(np.mean(task_ents)))

        # Everything else
        rewards_tensor = self.f_rewards(*all_input_values)
        returns_tensor = self.f_returns(*all_input_values)
        returns_tensor = np.squeeze(returns_tensor)  

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

        summary.value.add(tag='train/rewards', simple_value=float(aug_rewards[-1]))
        summary.value.add(tag='train/total_rewards', simple_value=float(np.sum(aug_rewards)))
        summary.value.add(tag='env/rewards', simple_value=float(env_rewards[-1]))
        summary.value.add(tag='env/total_rewards', simple_value=float(np.sum(env_rewards)))

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


        # Joint optimization of policy and task encoder
        logger.log("Computing loss before")
        self.optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        loss= self.optimizer.loss(all_input_values)
        logger.record_tabular('LossAfter', loss)
        logger.record_tabular('MeanKL', mean_kl)

        summary.value.add(tag='train/loss', simple_value=float(loss))
        summary.value.add(tag='train/mean_kl', simple_value=float(mean_kl))

        # Optimize trajectory encoder
        logger.log("Optimizing trajectory encoder...")
        logger.log("Optimizing")
        self.traj_enc_optimizer.optimize(all_input_values)
        logger.log("Computing KL after")
        traj_mean_kl = self.traj_enc_optimizer.constraint_val(all_input_values)
        logger.log("Computing loss after")
        traj_loss = self.traj_enc_optimizer.loss(all_input_values)
        logger.record_tabular('TrajEnc/LossAfter', traj_loss)
        logger.record_tabular('TrajEnc/MeanKL', traj_mean_kl)

        summary.value.add(tag='train/loss', simple_value=float(traj_loss))
        summary.value.add(tag='train/mean_kl', simple_value=float(traj_mean_kl))

        self.summary_writer.add_summary(summary, self.sampler.step)

        return loss, traj_loss, samples_data


    @overrides
    def optimize_policy(self, itr):
        paths = self.obtain_samples(itr)
        samples_data = self.process_samples(itr, paths)
        self.log_diagnostics(paths)

        loss, traj_loss, samples_data = self.optimize(samples_data) 
        # TODO: check the squeeze/dimension handling for both convolutions

        return self.get_itr_snapshot(itr, samples_data)
    

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
                params = self.optimize_policy(itr)
                if self.plot and itr > self.plot_warmup_itrs:
                    rollout(
                        self.env,
                        self.policy,
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
            trajectory_encoder=self.traj_encoder,
        )
