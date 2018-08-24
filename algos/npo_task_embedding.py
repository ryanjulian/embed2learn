from enum import Enum
from enum import unique
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from garage.core import Serializable
from garage.misc import ext
from garage.misc import special
from garage.misc.overrides import overrides
import garage.misc.logger as logger
from garage.tf.algos import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.misc.tensor_utils import compute_advantages
from garage.tf.misc.tensor_utils import discounted_returns
from garage.tf.misc.tensor_utils import filter_valids
from garage.tf.misc.tensor_utils import filter_valids_dict
from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import flatten_batch_dict
from garage.tf.misc.tensor_utils import flatten_inputs
from garage.tf.misc.tensor_utils import graph_inputs
from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.plotter import Plotter

from sandbox.embed2learn.embeddings import StochasticEmbedding
from sandbox.embed2learn.misc.metrics import rrse, rmse
from sandbox.embed2learn.policies import GaussianMLPMultitaskPolicy
from sandbox.embed2learn.policies import StochasticMultitaskPolicy
from sandbox.embed2learn.samplers import TaskEmbeddingSampler
from sandbox.embed2learn.samplers.task_embedding_sampler import rollout
from sandbox.embed2learn.tf.network_utils import softclip


@unique
class PGLoss(Enum):
    # VPG and TRPO
    VANILLA = "vanilla"
    # PPO
    CLIP = "clip"


class NPOTaskEmbedding(BatchPolopt, Serializable):
    """
    Natural Policy Optimization with Task Embeddings
    """

    def __init__(self,
                 name="NPOTaskEmbedding",
                 pg_loss=PGLoss.VANILLA,
                 kl_constraint=None,
                 optimizer=LbfgsOptimizer,
                 optimizer_args=dict(),
                 step_size=0.01,
                 num_minibatches=None,
                 num_opt_epochs=None,
                 policy=None,
                 policy_ent_coeff=1e-2,
                 embedding_ent_coeff=1e-2,
                 embedding_reg_coeff=0.,
                 inference=None,
                 inference_optimizer=LbfgsOptimizer,
                 inference_optimizer_args=dict(),
                 inference_ce_coeff=1e-3,
                 softplus_policy_ent=False,
                 softplus_embedding_ent=False,
                 stop_policy_ent_gradient=False,
                 logli_policy_ent=False,
                 logli_embedding_ent=False,
                 **kwargs):
        Serializable.quick_init(self, locals())
        assert kwargs['env'].task_space
        assert isinstance(policy, StochasticMultitaskPolicy)
        assert isinstance(inference, StochasticEmbedding)

        self.name = name
        self._name_scope = tf.name_scope(self.name)

        self._pg_loss = pg_loss
        self._policy_opt_inputs = None
        self._inference_opt_inputs = None
        self._softplus_policy_ent = softplus_policy_ent
        self._softplus_embedding_ent = softplus_embedding_ent
        self._stop_policy_ent_gradient = stop_policy_ent_gradient
        self._logli_policy_ent = logli_policy_ent
        self._logli_embedding_ent = logli_embedding_ent

        with tf.name_scope(self.name):
            # Optimizer for policy and embedding networks
            self.optimizer = optimizer(**optimizer_args)
            self.step_size = float(step_size)
            self.policy_ent_coeff = float(policy_ent_coeff)
            self.embedding_ent_coeff = float(embedding_ent_coeff)
            self.embedding_reg_coeff = float(embedding_reg_coeff)

            self.inference = inference
            self.inference_ce_coeff = inference_ce_coeff
            self.inference_optimizer = inference_optimizer(
                **inference_optimizer_args)

            sampler_cls = TaskEmbeddingSampler
            sampler_args = dict(inference=self.inference, )
            super(NPOTaskEmbedding, self).__init__(
                sampler_cls=sampler_cls,
                sampler_args=sampler_args,
                policy=policy,
                **kwargs)

    @overrides
    def start_worker(self, sess):
        self.sampler.start_worker()
        if self.plot:
            self.plotter = Plotter(
                self.env, self.policy, sess=sess, rollout=rollout)
            self.plotter.start()

    @overrides
    def init_opt(self):
        if self.policy.recurrent:
            raise NotImplementedError

        # Input variables
        (pol_loss_inputs, pol_opt_inputs, infer_loss_inputs,
         infer_opt_inputs) = self._build_inputs()

        self._policy_opt_inputs = pol_opt_inputs
        self._inference_opt_inputs = infer_opt_inputs

        # Jointly optimize policy and embedding network
        pol_loss, pol_kl, embed_kl = self._build_policy_loss(pol_loss_inputs)
        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy,
            leq_constraint=(pol_kl, self.step_size),
            inputs=flatten_inputs(self._policy_opt_inputs),
            constraint_name="mean_kl")

        # Optimize inference distribution separately (supervised learning)
        infer_loss, infer_kl = self._build_inference_loss(infer_loss_inputs)
        self.inference_optimizer.update_opt(
            loss=infer_loss,
            target=self.inference,
            inputs=flatten_inputs(self._inference_opt_inputs))

        return dict()

    #### Loss function network ################################################

    def _build_inputs(self):
        """
        Builds input variables (and trivial views thereof) for the loss
        function network
        """

        observation_space = self.policy.observation_space
        action_space = self.policy.action_space
        task_space = self.policy.task_space
        latent_space = self.policy.latent_space
        trajectory_space = self.inference.input_space

        policy_dist = self.policy._dist
        embed_dist = self.policy.embedding._dist
        infer_dist = self.inference._dist

        with tf.name_scope("inputs"):
            obs_var = observation_space.new_tensor_variable(
                'obs',
                extra_dims=1 + 1,
            )

            task_var = task_space.new_tensor_variable(
                'task',
                extra_dims=1 + 1,
            )

            action_var = action_space.new_tensor_variable(
                'action',
                extra_dims=1 + 1,
            )

            reward_var = tensor_utils.new_tensor(
                'reward',
                ndim=1 + 1,
                dtype=tf.float32,
            )

            latent_var = latent_space.new_tensor_variable(
                'latent',
                extra_dims=1 + 1,
            )

            baseline_var = tensor_utils.new_tensor(
                'baseline',
                ndim=1 + 1,
                dtype=tf.float32,
            )

            trajectory_var = trajectory_space.new_tensor_variable(
                'trajectory',
                extra_dims=1 + 1,
            )

            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="valid")

            # Policy state (for RNNs)
            policy_state_info_vars = {
                k: tf.placeholder(
                    tf.float32, shape=[None] * (1 + 1) + list(shape), name=k)
                for k, shape in self.policy.state_info_specs
            }
            policy_state_info_vars_list = [
                policy_state_info_vars[k] for k in self.policy.state_info_keys
            ]

            # Old policy distribution (for KL)
            policy_old_dist_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * (1 + 1) + list(shape),
                    name='policy_old_%s' % k)
                for k, shape in policy_dist.dist_info_specs
            }
            policy_old_dist_info_vars_list = [
                policy_old_dist_info_vars[k]
                for k in policy_dist.dist_info_keys
            ]

            # Embedding state (for RNNs)
            embed_state_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * (1 + 1) + list(shape),
                    name='embed_%s' % k)
                for k, shape in self.policy.embedding.state_info_specs
            }
            embed_state_info_vars_list = [
                embed_state_info_vars[k]
                for k in self.policy.embedding.state_info_keys
            ]

            # Old embedding distribution (for KL)
            embed_old_dist_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * (1 + 1) + list(shape),
                    name='embed_old_%s' % k)
                for k, shape in embed_dist.dist_info_specs
            }
            embed_old_dist_info_vars_list = [
                embed_old_dist_info_vars[k] for k in embed_dist.dist_info_keys
            ]

            # Inference distribution state (for RNNs)
            infer_state_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * (1 + 1) + list(shape),
                    name='infer_%s' % k)
                for k, shape in self.inference.state_info_specs
            }
            infer_state_info_vars_list = [
                infer_state_info_vars[k]
                for k in self.inference.state_info_keys
            ]

            # Old inference distribution (for KL)
            infer_old_dist_info_vars = {
                k: tf.placeholder(
                    tf.float32,
                    shape=[None] * (1 + 1) + list(shape),
                    name='infer_old_%s' % k)
                for k, shape in infer_dist.dist_info_specs
            }
            infer_old_dist_info_vars_list = [
                infer_old_dist_info_vars[k] for k in infer_dist.dist_info_keys
            ]

            # Flattened view
            with tf.name_scope("flat"):
                obs_flat = flatten_batch(obs_var, name="obs_flat")
                task_flat = flatten_batch(task_var, name="task_flat")
                action_flat = flatten_batch(action_var, name="action_flat")
                reward_flat = flatten_batch(reward_var, name="reward_flat")
                latent_flat = flatten_batch(latent_var, name="latent_flat")
                trajectory_flat = flatten_batch(
                    trajectory_var, name="trajectory_flat")
                valid_flat = flatten_batch(valid_var, name="valid_flat")
                policy_state_info_vars_flat = flatten_batch_dict(
                    policy_state_info_vars, name="policy_state_info_vars_flat")
                policy_old_dist_info_vars_flat = flatten_batch_dict(
                    policy_old_dist_info_vars,
                    name="policy_old_dist_info_vars_flat")
                embed_state_info_vars_flat = flatten_batch_dict(
                    embed_state_info_vars, name="embed_state_info_vars_flat")
                embed_old_dist_info_vars_flat = flatten_batch_dict(
                    embed_old_dist_info_vars,
                    name="embed_old_dist_info_vars_flat")
                infer_state_info_vars_flat = flatten_batch_dict(
                    infer_state_info_vars, name="infer_state_info_vars_flat")
                infer_old_dist_info_vars_flat = flatten_batch_dict(
                    infer_old_dist_info_vars,
                    name="infer_old_dist_info_vars_flat")

            # Valid view
            with tf.name_scope("valid"):
                action_valid = filter_valids(
                    action_flat, valid_flat, name="action_valid")
                policy_state_info_vars_valid = filter_valids_dict(
                    policy_state_info_vars_flat,
                    valid_flat,
                    name="policy_state_info_vars_valid")
                policy_old_dist_info_vars_valid = filter_valids_dict(
                    policy_old_dist_info_vars_flat,
                    valid_flat,
                    name="policy_old_dist_info_vars_valid")
                embed_old_dist_info_vars_valid = filter_valids_dict(
                    embed_old_dist_info_vars_flat,
                    valid_flat,
                    name="embed_old_dist_info_vars_valid")
                infer_old_dist_info_vars_valid = filter_valids_dict(
                    infer_old_dist_info_vars_flat,
                    valid_flat,
                    name="infer_old_dist_info_vars_valid")

        # Policy and embedding network loss and optimizer inputs
        pol_flat = graph_inputs(
            "PolicyLossInputsFlat",
            obs_var=obs_flat,
            task_var=task_flat,
            action_var=action_flat,
            reward_var=reward_flat,
            latent_var=latent_flat,
            trajectory_var=trajectory_flat,
            valid_var=valid_flat,
            policy_state_info_vars=policy_state_info_vars_flat,
            policy_old_dist_info_vars=policy_old_dist_info_vars_flat,
            embed_state_info_vars=embed_state_info_vars_flat,
            embed_old_dist_info_vars=embed_old_dist_info_vars_flat,
        )
        pol_valid = graph_inputs(
            "PolicyLossInputsValid",
            action_var=action_valid,
            policy_state_info_vars=policy_state_info_vars_valid,
            policy_old_dist_info_vars=policy_old_dist_info_vars_valid,
            embed_old_dist_info_vars=embed_old_dist_info_vars_valid,
        )
        policy_loss_inputs = graph_inputs(
            "PolicyLossInputs",
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            trajectory_var=trajectory_var,
            task_var=task_var,
            latent_var=latent_var,
            valid_var=valid_var,
            policy_state_info_vars=policy_state_info_vars,
            policy_old_dist_info_vars=policy_old_dist_info_vars,
            embed_state_info_vars=embed_state_info_vars,
            embed_old_dist_info_vars=embed_old_dist_info_vars,
            flat=pol_flat,
            valid=pol_valid,
        )
        # Special variant for the optimizer
        # * Uses lists instead of dicts for the distribution parameters
        # * Omits flats and valids
        # TODO: eliminate
        policy_opt_inputs = graph_inputs(
            "PolicyOptInputs",
            obs_var=obs_var,
            action_var=action_var,
            reward_var=reward_var,
            baseline_var=baseline_var,
            trajectory_var=trajectory_var,
            task_var=task_var,
            latent_var=latent_var,
            valid_var=valid_var,
            policy_state_info_vars_list=policy_state_info_vars_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_vars_list,
            embed_state_info_vars_list=embed_state_info_vars_list,
            embed_old_dist_info_vars_list=embed_old_dist_info_vars_list,
        )

        # Inference network loss and optimizer inputs
        infer_flat = graph_inputs(
            "InferenceLossInputsFlat",
            latent_var=latent_flat,
            trajectory_var=trajectory_flat,
            valid_var=valid_flat,
            infer_state_info_vars=infer_state_info_vars_flat,
            infer_old_dist_info_vars=infer_old_dist_info_vars_flat,
        )
        infer_valid = graph_inputs(
            "InferenceLossInputsValid",
            infer_old_dist_info_vars=infer_old_dist_info_vars_valid,
        )
        inference_loss_inputs = graph_inputs(
            "InferenceLossInputs",
            latent_var=latent_var,
            trajectory_var=trajectory_var,
            valid_var=valid_var,
            infer_state_info_vars=infer_state_info_vars,
            infer_old_dist_info_vars=infer_old_dist_info_vars,
            flat=infer_flat,
            valid=infer_valid,
        )
        # Special variant for the optimizer
        # * Uses lists instead of dicts for the distribution parameters
        # * Omits flats and valids
        # TODO: eliminate
        inference_opt_inputs = graph_inputs(
            "InferenceOptInputs",
            latent_var=latent_var,
            trajectory_var=trajectory_var,
            valid_var=valid_var,
            infer_state_info_vars_list=infer_state_info_vars_list,
            infer_old_dist_info_vars_list=infer_old_dist_info_vars_list,
        )

        return (policy_loss_inputs, policy_opt_inputs, inference_loss_inputs,
                inference_opt_inputs)

    def _build_policy_loss(self, i):
        """ Build policy network loss """
        pol_dist = self.policy._dist

        # Entropy terms
        (embedding_entropy, embedding_divergence, inference_ce,
            policy_entropy) = self._build_entropy_terms(i)

        # Augment the path rewards with entropy terms
        with tf.name_scope("augmented_rewards"):
            rewards = i.reward_var \
                      - (self.inference_ce_coeff * inference_ce) \
                      + (self.policy_ent_coeff * policy_entropy)

        with tf.name_scope("policy_loss"):
            with tf.name_scope("advantages"):
                advantages = compute_advantages(
                    self.discount,
                    self.gae_lambda,
                    self.max_path_length,
                    i.baseline_var,
                    rewards,
                    name="advantages"
                )

                # Flatten and filter valids
                adv_flat = flatten_batch(advantages, name="adv_flat")
                adv_valid = filter_valids(
                    adv_flat, i.flat.valid_var, name="adv_valid")

            policy_dist_info_flat = self.policy.dist_info_sym(
                i.flat.task_var,
                i.flat.obs_var,
                i.flat.policy_state_info_vars,
                name="policy_dist_info_flat")
            policy_dist_info_valid = filter_valids_dict(
                policy_dist_info_flat,
                i.flat.valid_var,
                name="policy_dist_info_valid")

            # Optionally normalize advantages
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

            # Calculate loss function and KL divergence
            with tf.name_scope("kl"):
                kl = pol_dist.kl_sym(
                    i.valid.policy_old_dist_info_vars,
                    policy_dist_info_valid,
                )
                pol_mean_kl = tf.reduce_mean(kl)

            # Calculate surrogate loss
            with tf.name_scope("surr_loss"):
                lr = pol_dist.likelihood_ratio_sym(
                    i.valid.action_var,
                    i.valid.policy_old_dist_info_vars,
                    policy_dist_info_valid,
                    name="lr")

                # Policy gradient surrogate objective
                surr_vanilla = lr * adv_valid

                if self._pg_loss == PGLoss.VANILLA:
                    # VPG, TRPO use the standard surrogate objective
                    surr_obj = tf.identity(surr_vanilla, name="surr_obj")
                elif self._pg_loss == PGLoss.CLIP:
                    # PPO uses a surrogate objective with clipped LR
                    lr_clip = tf.clip_by_value(
                        lr,
                        1. - self.step_size,
                        1. + self.step_size,
                        name="lr_clip")
                    surr_clip = lr_clip * adv_valid
                    surr_obj = tf.minimum(
                        surr_vanilla, surr_clip, name="surr_obj")
                else:
                    raise NotImplementedError("Unknown PGLoss")

                # Maximize E[surrogate objective] by minimizing
                # -E_t[surrogate objective]
                surr_loss = -tf.reduce_mean(surr_obj)

                # Embedding entropy bonus
                surr_loss -= self.embedding_ent_coeff * embedding_entropy
                # Embedding divegence penalty
                surr_loss += self.embedding_reg_coeff * embedding_divergence

            embed_mean_kl = self._build_embedding_kl(i)

        # Diagnostic functions
        self.f_policy_kl = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            pol_mean_kl,
            log_name="f_policy_kl")

        self.f_rewards = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            rewards,
            log_name="f_rewards")

        # returns = self._build_returns(rewards)
        returns = discounted_returns(self.discount, self.max_path_length,
                                     rewards, name="returns")
        self.f_returns = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            returns,
            log_name="f_returns")

        return surr_loss, pol_mean_kl, embed_mean_kl

    def _build_entropy_terms(self, i):
        """ Calculate entropy terms """

        with tf.name_scope("entropy_terms"):
            # 1. Embedding distribution total entropy
            with tf.name_scope('embedding_entropy'):
                task_dim = self.policy.task_space.flat_dim
                all_task_one_hots = tf.one_hot(
                    np.arange(task_dim), task_dim, name="all_task_one_hots")

                if not self._logli_embedding_ent:
                    all_task_entropies = self.policy.embedding.entropy_sym(
                        all_task_one_hots)
                    all_task_entropies = softclip(all_task_entropies, 0., 5.)
                else:
                    all_task_latent_dist_infos = (
                        self.policy.embedding.dist_info_sym(all_task_one_hots))
                    all_task_latent_means = all_task_latent_dist_infos["mean"]
                    all_latent_loglis = (
                        self.policy.embedding.log_likelihood_sym(
                            all_task_one_hots,
                            all_task_latent_means,
                            name="all_latent_loglis",
                        )
                    )
                    # Clip extreme log likelihoods to prevent exploding
                    # gradients
                    all_latent_loglis = softclip(
                        all_latent_loglis, np.log(1e-2), 0.)
                    all_task_entropies = -all_latent_loglis

                embedding_entropy = tf.reduce_mean(
                    all_task_entropies, name="embedding_entropy")

            # 2. Embedding distribution L-inf norm (regularization)
            mean_std = self.policy.embedding.dist_info_sym(all_task_one_hots)
            all_task_means = mean_std["mean"]
            embedding_divergence = tf.norm(all_task_means, ord=np.inf)

            # 3. Inference distribution cross-entropy (log-likelihood)
            with tf.name_scope('inference_ce'):
                traj_ll_flat = self.inference.log_likelihood_sym(
                    i.flat.trajectory_var,
                    self.policy._embedding.latent_sym(i.flat.task_var),
                    name="traj_ll_flat")
                traj_ll = tf.reshape(
                    traj_ll_flat, [-1, self.max_path_length], name="traj_ll")
                # Softclip extreme log likelihoods to prevent exploding
                # gradients
                traj_ll = softclip(traj_ll, np.log(1e-3), 0.)
                inference_ce = -traj_ll

            # 4. Policy path entropies
            with tf.name_scope('policy_entropy'):
                if not self._logli_policy_ent:
                    policy_entropy_flat = self.policy.entropy_sym(
                        i.task_var, i.obs_var, name="policy_entropy_flat")
                    policy_entropy = tf.reshape(
                        policy_entropy_flat, [-1, self.max_path_length],
                        name="policy_entropy")
                else:
                    action_loglis = self.policy.log_likelihood_sym(
                        i.task_var,
                        i.obs_var,
                        i.action_var,
                        name="action_loglis",
                    )
                    policy_entropy = -action_loglis

            # optionally add nonlinearities to entropy terms
            if self._softplus_policy_ent:
                policy_entropy = tf.nn.softplus(policy_entropy)

            if self._softplus_embedding_ent:
                embedding_entropy = tf.nn.softplus(embedding_entropy)

            # optionally stop policy entropy gradients
            if self._stop_policy_ent_gradient:
                policy_entropy = tf.stop_gradient(policy_entropy)

        # Diagnostic functions
        self.f_task_entropies = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            all_task_entropies,
            log_name="f_task_entropies")
        self.f_embedding_entropy = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            embedding_entropy,
            log_name="f_embedding_entropy")
        self.f_embedding_divergence = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            embedding_divergence,
            log_name="f_embedding_entropy")
        self.f_inference_ce = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            tf.reduce_mean(inference_ce * i.valid_var),
            log_name="f_inference_ce")
        self.f_policy_entropy = tensor_utils.compile_function(
            flatten_inputs(self._policy_opt_inputs),
            tf.reduce_mean(policy_entropy * i.valid_var),
            log_name="f_policy_entropy")

        return (
            embedding_entropy,
            embedding_divergence,
            inference_ce,
            policy_entropy,
        )

    def _build_embedding_kl(self, i):
        dist = self.policy._embedding._dist
        with tf.name_scope("embedding_kl"):
            # new distribution
            embed_dist_info_flat = self.policy._embedding.dist_info_sym(
                i.flat.task_var,
                i.flat.embed_state_info_vars,
                name="embed_dist_info_flat")
            embed_dist_info_valid = filter_valids_dict(
                embed_dist_info_flat,
                i.flat.valid_var,
                name="embed_dist_info_valid")

            # calculate KL divergence
            kl = dist.kl_sym(i.valid.embed_old_dist_info_vars,
                             embed_dist_info_valid)
            mean_kl = tf.reduce_mean(kl)

            # Diagnostic function
            self.f_embedding_kl = tensor_utils.compile_function(
                flatten_inputs(self._policy_opt_inputs),
                mean_kl,
                log_name="f_embedding_kl")

            return mean_kl

    def _build_inference_loss(self, i):
        """ Build loss function for the inference network """

        infer_dist = self.inference._dist
        with tf.name_scope("infer_loss"):
            traj_ll_flat = self.inference.log_likelihood_sym(
                i.flat.trajectory_var, i.flat.latent_var, name="traj_ll_flat")
            traj_ll = tf.reshape(
                traj_ll_flat, [-1, self.max_path_length], name="traj_ll")

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
                i.flat.valid_var,
                name="discount_traj_ll_valid")

            with tf.name_scope("loss"):
                infer_loss = -tf.reduce_mean(
                    discount_traj_ll_valid, name="infer_loss")

            with tf.name_scope("kl"):
                # Calculate predicted embedding distributions for each timestep
                infer_dist_info_flat = self.inference.dist_info_sym(
                    i.flat.trajectory_var,
                    i.flat.infer_state_info_vars,
                    name="infer_dist_info_flat")

                infer_dist_info_valid = filter_valids_dict(
                    infer_dist_info_flat,
                    i.flat.valid_var,
                    name="infer_dist_info_valid")

                # Calculate KL divergence
                kl = infer_dist.kl_sym(i.valid.infer_old_dist_info_vars,
                                       infer_dist_info_valid)
                infer_kl = tf.reduce_mean(kl, name="infer_kl")

            return infer_loss, infer_kl

    #### Sampling and training ################################################
    def _policy_opt_input_values(self, samples_data):
        """ Map rollout samples to the policy optimizer inputs """

        policy_state_info_list = [
            samples_data["agent_infos"][k] for k in self.policy.state_info_keys
        ]
        policy_old_dist_info_list = [
            samples_data["agent_infos"][k]
            for k in self.policy._dist.dist_info_keys
        ]
        embed_state_info_list = [
            samples_data["latent_infos"][k]
            for k in self.policy.embedding.state_info_keys
        ]
        embed_old_dist_info_list = [
            samples_data["latent_infos"][k]
            for k in self.policy.embedding._dist.dist_info_keys
        ]
        policy_opt_input_values = self._policy_opt_inputs._replace(
            obs_var=samples_data["observations"],
            action_var=samples_data["actions"],
            reward_var=samples_data["rewards"],
            baseline_var=samples_data["baselines"],
            trajectory_var=samples_data["trajectories"],
            task_var=samples_data["tasks"],
            latent_var=samples_data["latents"],
            valid_var=samples_data["valids"],
            policy_state_info_vars_list=policy_state_info_list,
            policy_old_dist_info_vars_list=policy_old_dist_info_list,
            embed_state_info_vars_list=embed_state_info_list,
            embed_old_dist_info_vars_list=embed_old_dist_info_list,
        )
        return flatten_inputs(policy_opt_input_values)

    def _inference_opt_input_values(self, samples_data):
        """ Map rollout samples to the inference optimizer inputs """

        infer_state_info_list = [
            samples_data["trajectory_infos"][k]
            for k in self.inference.state_info_keys
        ]
        infer_old_dist_info_list = [
            samples_data["trajectory_infos"][k]
            for k in self.inference._dist.dist_info_keys
        ]
        inference_opt_input_values = self._inference_opt_inputs._replace(
            latent_var=samples_data["latents"],
            trajectory_var=samples_data["trajectories"],
            valid_var=samples_data["valids"],
            infer_state_info_vars_list=infer_state_info_list,
            infer_old_dist_info_vars_list=infer_old_dist_info_list,
        )

        return flatten_inputs(inference_opt_input_values)

    def evaluate(self, policy_opt_input_values, samples_data):
        # Everything else
        rewards_tensor = self.f_rewards(*policy_opt_input_values)
        returns_tensor = self.f_returns(*policy_opt_input_values)
        returns_tensor = np.squeeze(returns_tensor)  # TODO
        # TODO: check the squeeze/dimension handling for both convolutions

        paths = samples_data['paths']
        valids = samples_data['valids']
        baselines = [path['baselines'] for path in paths]
        env_rewards = [path['rewards'] for path in paths]
        env_rewards = tensor_utils.concat_tensor_list(env_rewards.copy())
        env_returns = [path['returns'] for path in paths]
        env_returns = tensor_utils.concat_tensor_list(env_returns.copy())
        env_average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

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
        d_rewards = np.mean(aug_rewards - env_rewards)
        logger.record_tabular('Policy/EntRewards', d_rewards)

        aug_average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])
        d_returns = np.mean(aug_average_discounted_return -
                            env_average_discounted_return)
        logger.record_tabular('Policy/EntReturns', d_returns)

        # Calculate explained variance
        ev = special.explained_variance_1d(
            np.concatenate(baselines), aug_returns)
        logger.record_tabular('Baseline/ExplainedVariance', ev)

        inference_rmse = (samples_data['trajectory_infos']['mean'] -
                          samples_data['latents'])**2.
        inference_rmse = np.sqrt(inference_rmse.mean())
        logger.record_tabular('Inference/RMSE', inference_rmse)

        inference_rrse = rrse(
            samples_data['latents'], samples_data['trajectory_infos']['mean'])
        logger.record_tabular('Inference/RRSE', inference_rrse)

        embed_ent = self.f_embedding_entropy(*policy_opt_input_values)
        logger.record_tabular('Embedding/Entropy', embed_ent)

        embed_div = self.f_embedding_divergence(*policy_opt_input_values)
        logger.record_tabular('Embedding/Divergence', embed_div)

        infer_ce = self.f_inference_ce(*policy_opt_input_values)
        logger.record_tabular('Inference/CrossEntropy', infer_ce)

        pol_ent = self.f_policy_entropy(*policy_opt_input_values)
        logger.record_tabular('Policy/Entropy', pol_ent)

        task_ents = self.f_task_entropies(*policy_opt_input_values)
        tasks = samples_data["tasks"][:, 0, :]
        _, task_indices = np.nonzero(tasks)
        path_lengths = np.sum(samples_data["valids"], axis=1)
        for t in range(self.policy.task_space.flat_dim):
            lengths = path_lengths[task_indices == t]
            completed = lengths < self.max_path_length
            pct_completed = np.mean(completed)
            num_samples = np.sum(lengths)
            num_trajs = lengths.shape[0]
            logger.record_tabular('Tasks/EpisodeLength/t={}'.format(t),
                                  np.mean(lengths))
            logger.record_tabular('Tasks/CompletionRate/t={}'.format(t),
                                  pct_completed)
            # logger.record_tabular('Tasks/NumSamples/t={}'.format(t),
            #                       num_samples)
            # logger.record_tabular('Tasks/NumTrajs/t={}'.format(t), num_trajs)
            logger.record_tabular('Tasks/Entropy/t={}'.format(t), task_ents[t])

        return samples_data

    def visualize_distribution(self, samples_data):
        """ Visualize embedding distribution """

        # distributions
        num_tasks = self.policy.task_space.flat_dim
        all_tasks = np.eye(num_tasks, num_tasks)
        _, latent_infos = self.policy._embedding.get_latents(all_tasks)
        for i in range(self.policy.latent_space.flat_dim):
            log_stds = latent_infos["log_std"][:, i]
            if self.policy.embedding._std_parameterization == "exp":
                stds = np.exp(log_stds)
            elif self.policy.embedding._std_parameterization == "softplus":
                stds = np.log(1. + log_stds)
            else:
                raise NotImplementedError
            logger.record_histogram_by_type(
                "normal",
                shape=[1000, num_tasks],
                key="Embedding/i={}".format(i),
                mean=latent_infos["mean"][:, i],
                stddev=stds)

        num_traj = self.batch_size // self.max_path_length
        # # samples
        # latents = samples_data["latents"][:num_traj, 0]
        # for i in range(self.policy.latent_space.flat_dim):
        #     logger.record_histogram("Embedding/samples/i={}".format(i),
        #                             latents[:, i])

        # action distributions
        actions = samples_data["actions"][:num_traj, ...]
        logger.record_histogram("Actions", actions)

    def train_policy_and_embedding_networks(self, policy_opt_input_values):
        """ Joint optimization of policy and embedding networks """

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(policy_opt_input_values)

        logger.log("Computing KL before")
        policy_kl_before = self.f_policy_kl(*policy_opt_input_values)
        embed_kl_before = self.f_embedding_kl(*policy_opt_input_values)

        logger.log("Optimizing")
        self.optimizer.optimize(policy_opt_input_values)

        logger.log("Computing KL after")
        policy_kl = self.f_policy_kl(*policy_opt_input_values)
        embed_kl = self.f_embedding_kl(*policy_opt_input_values)

        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(policy_opt_input_values)

        logger.record_tabular('Policy/LossBefore', loss_before)
        logger.record_tabular('Policy/LossAfter', loss_after)
        logger.record_tabular('Policy/KLBefore', policy_kl_before)
        logger.record_tabular('Policy/KL', policy_kl)
        logger.record_tabular('Policy/dLoss', loss_before - loss_after)
        logger.record_tabular('Embedding/KLBefore', embed_kl_before)
        logger.record_tabular('Embedding/KL', embed_kl)

        return loss_after

    def train_inference_network(self, inference_opt_input_values):
        """ Optimize inference network """

        logger.log("Optimizing inference network...")
        infer_loss_before = self.inference_optimizer.loss(
            inference_opt_input_values)
        logger.record_tabular('Inference/Loss', infer_loss_before)
        self.inference_optimizer.optimize(inference_opt_input_values)
        infer_loss_after = self.inference_optimizer.loss(
            inference_opt_input_values)
        logger.record_tabular('Inference/dLoss',
                              infer_loss_before - infer_loss_after)

        return infer_loss_after

    @overrides
    def optimize_policy(self, itr, **kwargs):
        paths = self.obtain_samples(itr)
        samples_data = self.process_samples(itr, paths)
        self.log_diagnostics(paths)

        policy_opt_input_values = self._policy_opt_input_values(samples_data)
        inference_opt_input_values = self._inference_opt_input_values(
            samples_data)

        self.train_policy_and_embedding_networks(policy_opt_input_values)
        self.train_inference_network(inference_opt_input_values)

        samples_data = self.evaluate(policy_opt_input_values, samples_data)
        self.visualize_distribution(samples_data)

        # Fit baseline
        logger.log("Fitting baseline...")
        if hasattr(self.baseline, 'fit_with_samples'):
            self.baseline.fit_with_samples(paths, samples_data)
        else:
            self.baseline.fit(paths)

        return self.get_itr_snapshot(itr, samples_data)

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
            with logger.prefix('itr #%d | ' % itr):
                params = self.optimize_policy(itr, )
                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                              "continue...")
                logger.log("Saving snapshot...")
                logger.save_itr_params(itr, params)
                logger.log("Saved")
                logger.record_tabular('IterTime', time.time() - itr_start_time)
                logger.record_tabular('Time', time.time() - start_time)
                logger.dump_tabular()
        self.shutdown_worker()
        if created_session:
            sess.close()

    @overrides
    def get_itr_snapshot(self, itr, _samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            # baseline=self.baseline,
            env=self.env,
            inference=self.inference,
        )
