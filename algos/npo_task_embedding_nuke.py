import time

import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import ext
from garage.misc.overrides import overrides
from garage.tf.algos import BatchPolopt
from garage.tf.optimizers import FirstOrderOptimizer
from garage.tf.optimizers import ConjugateGradientOptimizer

from sandbox.embed2learn.algos.utils import flatten_batch, filter_valids
from sandbox.embed2learn.embeddings import StochasticMultitaskPolicy
from sandbox.embed2learn.embeddings import StochasticEmbedding
from sandbox.embed2learn.samplers.task_embedding_sampler import TaskEmbeddingSampler


def _optimizer_or_default(optimizer, args):
    use_optimizer = optimizer
    use_args = args
    if use_optimizer is None:
        if use_args is None:
            use_args = dict()
        use_optimizer = ConjugateGradientOptimizer(**use_args)
    return use_optimizer


class NPOTaskEmbedding(BatchPolopt, Serializable):

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

        self.name = name
        self._name_scope = tf.name_scope(self.name)

        with tf.name_scope(self.name):
            self.optimizer = _optimizer_or_default(optimizer, optimizer_args)
            self.step_size = float(step_size)
            self.policy_ent_coeff = float(policy_ent_coeff)
            self.task_enc_ent_coeff = float(task_encoder_ent_coeff)

            self.traj_encoder = trajectory_encoder
            self.traj_enc_ent_coeff = trajectory_encoder_ent_coeff
            self.traj_enc_optimizer = FirstOrderOptimizer(
                tf.train.AdamOptimizer,
                dict(learning_rate=float(trajectory_encoder_learning_rate)))

            sampler_cls = TaskEmbeddingSampler
            sampler_args = dict(trajectory_encoder=self.traj_encoder, )

            super(NPOTaskEmbedding, self).__init__(
                sampler_cls=sampler_cls,
                sampler_args=sampler_args,
                policy=policy,
                **kwargs)

    def init_opt(self):
        pol_loss, pol_mean_kl, traj_enc_loss, pol_inputs = self._build_opt()

        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy,
            leq_constraint=(pol_mean_kl, self.step_size),
            inputs=pol_inputs,
            constraint_name="mean_kl",
        )

    def _build_opt(self):
        with tf.name_scope("build_opt"):
            is_recurrent = self.policy.recurrent
            if is_recurrent:
                raise NotImplementedError

            self._obs_ph = self.policy._obs_ph
            self._task_ph = self.policy.task_ph
            self._latent_bp = self.policy.bp_latents

            self._latents_ph = tf.placeholder(tf.float32, shape=[None, self.policy.embedding.latent_space.flat_dim], name='traj_enc_latent_ph')

            self._action_ph = tf.placeholder(tf.float32, shape=[None, self.env.action_space.flat_dim], name="action_ph")
            self._baseline_ph = tf.placeholder(tf.float32, shape=[None, None], name='baseline_ph')
            self._trajectory_ph = self.traj_encoder.input_ph
            self._valid_ph = tf.placeholder(tf.float32, shape=[None, None], name='valid_ph')
            self._reward_ph = tf.placeholder(tf.float32, shape=[None, None], name='reward_ph')

            self._dist_info_vars = {'mean': self.policy.dist.means, 'log_stds': self.policy.dist.log_stds}
            dist_type = type(self.policy.dist)

            self._old_dist_mean = tf.placeholder(tf.float32, shape=[None, self.env.action_space.flat_dim], name="old_dist_mean_ph")
            self._old_dist_stds = tf.placeholder(tf.float32, shape=[None, self.env.action_space.flat_dim], name="old_dist_stds_ph")
            self._old_dist = dist_type(self._old_dist_mean, self._old_dist_stds, self.policy.dist.dim)
            self._old_dist_vars_list = [self._old_dist.means, self._old_dist.log_stds]

            surr_loss, pol_mean_kl, traj_enc_loss, rewards = self.build_loss()

            self._surr_loss = surr_loss
            self._pol_mean_kl = pol_mean_kl,
            self._traj_enc_loss = traj_enc_loss
            self._rewards = rewards

            policy_input_list = self._build_policy_input()

        return self._surr_loss, self._pol_mean_kl, self._traj_enc_loss, policy_input_list

    def _build_policy_input(self):
        input_list = [
            self._obs_ph,
            self._action_ph,
            self._reward_ph,
            self._reward_ph,
            self._baseline_ph,
            self._trajectory_ph,
            self._task_ph,
            self._latents_ph,  # TODO check if we actually need this for policy+embedding optimization
            self._valid_ph,
        ] + self._old_dist_vars_list
        return input_list


    def build_loss(self):
        task_enc_entropy, traj_ll, pol_entropy = self.get_entropy()
        rewards = self.get_rewards(task_enc_entropy, traj_ll, pol_entropy)
        surr_loss, pol_mean_kl = self.get_task_loss(task_enc_entropy, traj_ll,
                                                    pol_entropy, rewards)
        traj_enc_loss = self.get_traj_loss(task_enc_entropy, traj_ll, pol_entropy)
        return surr_loss, pol_mean_kl, traj_enc_loss, rewards

    def get_entropy(self):

        traj_flat = flatten_batch(self._trajectory_ph, name="traj_flat")
        latent_flat = flatten_batch(self._latent_bp)

        with tf.name_scope("task_encoder_entropy"):
            all_task_entropies = self.policy.embedding.entropy()
            task_enc_entropy = tf.reduce_mean(
                all_task_entropies, name="task_enc_entropy")

        with tf.name_scope("traj_encoder_ce"):
            traj_ll_flat = self.traj_encoder._dist.log_prob(self._latents_ph)  #TODO switch to a new sample tensor or latents history
            traj_ll = tf.reshape(
                traj_ll_flat, [-1, self.max_path_length], name="traj_ll")

        with tf.name_scope('policy_entropy'):
            pol_entropy_flat = self.policy.dist.entropy(name="pol_entropy_flat")
            pol_entropy = tf.reshape(
                pol_entropy_flat, [-1, self.max_path_length],
                name="pol_entropy")

        return task_enc_entropy, traj_ll, pol_entropy

    def get_rewards(self, task_encoder_entropy, traj_ll, pol_entropy):
        with tf.name_scope("rewards"):
            rewards = self._reward_ph + \
                      (self.traj_enc_ent_coeff * traj_ll) + \
                      (self.policy_ent_coeff * pol_entropy)
        return rewards

    def get_task_loss(self, task_enc_entropy, traj_ll, pol_entropy, rewards):
        dist = self.policy._dist
        act_flat = flatten_batch(self._action_ph, name="act_flat")
        valid_flat = flatten_batch(self._valid_ph, name="valid_flat")
        # print(self._old_dist_vars)
        # old_dist_info_flat = flatten_batch_dict(
        #     self._old_dist_vars, name="old_dist_info_flat")
        # print(valid_flat)

        with tf.name_scope("policy_loss"):
            with tf.name_scope("advantages"):
                gamma_lambda = tf.constant(
                    float(self.discount) * float(self.gae_lambda),
                    dtype=tf.float32,
                    shape=[self.max_path_length, 1, 1])

                advantage_filter = tf.cumprod(gamma_lambda, exclusive=True)
                pad = tf.zeros_like(self._baseline_ph[:, :1])
                baseline_shift = tf.concat([self._baseline_ph[:, 1:], pad], 1)
                deltas = rewards + \
                         (self.discount * baseline_shift) - \
                         self._baseline_ph
                # Convolve deltas with the discount filter to get advantages
                deltas_pad = tf.expand_dims(
                    tf.concat([deltas, tf.zeros_like(deltas[:, :-1])], axis=1),
                    axis=2)
                adv = tf.nn.conv1d(
                    deltas_pad, advantage_filter, stride=1, padding='VALID')
                advantages = tf.reshape(adv, [-1])

            adv_flat = flatten_batch(advantages, name="adv_flat")
            action_valid = filter_valids(
                act_flat, valid_flat, name="action_valid")
            # old_dist_info_vars_valid = filter_valids_dict(
            #     old_dist_info_flat,
            #     valid_flat,
            #     name="old_dist_info_vars_valid")
            self._old_dist.flatten_valid_filter(valid_flat)

            adv_valid = filter_valids(adv_flat, valid_flat, name="adv_valid")

            dist.flatten_valid_filter(valid_flat)
            eps = tf.constant(1e-8, dtype=tf.float32)
            if self.center_adv:
                with tf.name_scope("center_adv"):
                    mean, var = tf.nn.moments(adv_valid, axes=[0])
                    adv_valid = tf.nn.batch_normalization(
                        adv_valid, mean, var, 0, 1, eps)
            if self.positive_adv:
                with tf.name_scope("positive_adv"):
                    m = tf.reduce_min(adv_valid)
                    adv_valid = (adv_valid - m) + eps

            with tf.name_scope("kl"):
                kl = dist.flat_kl_divergence(self._old_dist)
                pol_mean_kl = tf.reduce_mean(kl)

            with tf.name_scope("surr_loss"):
                lr = dist.log_prob_ratio(
                    action_valid,
                    self._old_dist,
                    valid_filter=True,
                    name="lr")
                surr_loss = -tf.reduce_mean(lr * adv_valid) - \
                            (self.task_enc_ent_coeff * task_enc_entropy)

        return surr_loss, pol_mean_kl

    def get_traj_loss(self, task_encoder_entropy, traj_ll, pol_entropy):
        return None

    def build_returns(self, rewards):
        with tf.name_scope("returns"):
            gamma = tf.constant(
                float(self.discount),
                dtype=tf.float32,
                shape=[self.max_path_length, 1, 1]
            )
            return_filter = tf.cumprod(gamma, exclusive=True)
            rewards_pad = tf.expand_dims(tf.concat([rewards, tf.zeros_like(rewards[:, :-1])], axis=1),
                                         axis=2)
            returns = tf.nn.conv1d(
                rewards_pad, return_filter, stride=1, padding='VALID')
            return returns

    @overrides
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
            self.optimize_policy(itr, )

        self.shutdown_worker()
        if created_session:
            sess.close()

    @overrides
    def optimize_policy(self, itr, **kwargs):
        paths = self.obtain_samples(itr)
        samples_data = self.process_samples(itr, paths)
        samples_data = self.optimize(samples_data)
        return samples_data

    def get_training_input(self, samples_data):
        policy_input_values = tuple(
            ext.extract(samples_data, 'observations', 'actions', 'rewards',
                        'baselines', 'trajectories', 'tasks', 'latents',
                        'valids'))
        agent_infos = samples_data["agent_infos"]
        # state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]  # TODO fix state info
        dist_info_list = [
            agent_infos[k] for k in self.policy._dist.dist_info_keys
        ]

        # policy_input_values += tuple(state_info_list) + tuple(dist_info_list)
        policy_input_values += tuple(dist_info_list)  # It might be old dist TODO check this ..

        latent_infos = samples_data['latent_info']

        # task_enc_state_info_list = [
        #     latent_infos[k] for k in self.policy.embedding.state_info_keys
        # ]
        task_enc_dist_info_list = [
            latent_infos[k] for k in self.policy.embedding._dist.dist_info_keys
        ]
        # policy_input_values += tuple(task_enc_state_info_list) + tuple(
        #     task_enc_dist_info_list)
        policy_input_values += tuple(task_enc_dist_info_list)


        # add trajectory encoder params
        trajectory_infos = samples_data["trajectory_infos"]
        traj_enc_state_info_list = [
            trajectory_infos[k] for k in self.traj_encoder.state_info_keys
        ]
        traj_enc_dist_info_list = [
            trajectory_infos[k] for k in self.traj_encoder._dist.dist_info_keys
        ]
        policy_input_values += tuple(traj_enc_state_info_list) + tuple(
            traj_enc_dist_info_list)

        traj_enc_input_values = [
            samples_data['trajectories'], samples_data['latents'],
            samples_data['valids']
        ]

        return policy_input_values, traj_enc_input_values

    def optimize(self, samples_data):
        policy_input_values, traj_enc_input_values = self.get_training_input(
            samples_data)

        self.train_task(policy_input_values)

        return None

    def get_feeds(self, samples_data):
        agent_infos = samples_data['agent_infos']
        trajectory_infos = samples_data['trajectory_infos']

    def train_task(self, all_input_values):
        pass

    def build_policy_input(self):
        input_list = [
            self._obs_ph,
            self._action_ph,
            self._reward_ph,
            self._baseline_ph,
            self._trajectory_ph,
            self._task_ph,
            self._latents_ph,     # TODO change it to one hot
            self._valid_ph,
        ]
        return input_list


