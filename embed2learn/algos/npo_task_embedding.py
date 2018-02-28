import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from rllab.sampler.utils import rollout

from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.misc import tensor_utils

from embed2learn.envs.multi_task_env import MultiTaskEnv
from embed2learn.embeddings.base import Embedding


def _optimizer_or_default(optimizer, args):
    use_optimizer = optimizer
    use_args = args
    if use_optimizer is None:
        if use_args is None:
            use_args = dict(name="optimizer")
        use_optimizer = PenaltyLbfgsOptimizer(**use_args)
    return use_optimizer


class TaskEmbeddingSampler(BatchSampler):
    def __init__(self,
                 *args,
                 task_encoder=None,
                 trajectory_encoder=None,
                 **kwargs):
        super(TaskEmbeddingSampler, self).__init__(*args, **kwargs)
        self.task_encoder = task_encoder
        self.traj_encoder = traj_encoder

    # TODO: don't call _get_scope_G()
    # TODO: custom rollout
    def _worker_collect_one_path(self, G, max_path_length, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        path = rollout(G.env, G.policy, max_path_length)
        return path, len(path["rewards"])

    def sample_paths(self,
                     policy_params,
                     max_samples,
                     max_path_legnth,
                     env_params=None,
                     scope=None):
        pool = parallel_sampler.singleton_pool
        pool.run_each(
            parallel_sampler._worker_set_policy_params,
            [(policy_params, scope)] * pool.n_parallel,
        )
        if env_params:
            pool.run_each(
                parallel_sampler._worker_set_env_params,
                [(env_params, scope)] * pool.n_parallel,
            )

        return pool.run_collect(
            self._worker_collect_one_path,
            threshold=max_samples,
            args=(max_path_length, scope),
            show_prog_bar=True,
        )

    def obtain_samples(self, itr):
        policy_params = self.algo.policy.get_param_values()
        env_params = self.algo.env.get_param_values()
        paths = self.sample_paths(
            policy_params=policy_params,
            env_params=env_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(
                paths, self.algo.batch_size)
            return paths_truncated

    #TODO: calculate trajectory latents
    def process_samples(self, itr, paths):
        return super(TaskEmbeddingSampler, self).process_samples(itr, paths)

    #TODO: embedding-specific diagnostics
    def log_diagnostics(self, paths):
        return super(TaskEmbeddingSampler, self).log_diagnostics(paths)


class NPOTaskEmbedding(BatchPolopt, Serializable):
    """
    Natural Policy Optimization with Task Embeddings
    """

    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 step_size=0.01,
                 task_encoder=None,
                 task_encoder_optimizer=None,
                 task_encoder_optimizer_args=None,
                 task_encoder_step_size=0.01,
                 trajectory_encoder=None,
                 trajectory_encoder_optimizer=None,
                 trajectory_encoder_optimizer_args=None,
                 trajectory_encoder_step_size=0.01,
                 **kwargs):
        Serializable.quick_init(self, locals())
        assert kwargs['env'].task_space
        assert isinstance(task_encoder, Embedding)
        assert isinstance(trajectory_encoder, Embedding)

        self.task_encoder = task_encoder
        self.traj_encoder = trajectory_encoder

        # Policy optimizer
        self.optimizer = _optimizer_or_default(optimizer, optimizer_args)
        self.step_size = step_size

        # Task encoder optimizer
        self.task_encoder_optimizer = _optimizer_or_default(
            task_encoder_optimizer, task_encoder_optimizer_args)
        self.task_encoder_step_size = task_encoder_step_size

        # Trajectory encoder optimizer
        self.traj_encoder_optimizer = _optimizer_or_default(
            trajectory_encoder_optimizer, trajectory_encoder_optimizer_args)
        self.traj_encoder_step_size = trajectory_encoder_step_size

        sampler_cls = TaskEmbeddingSampler
        sampler_args = dict(
            task_encoder=self.task_encoder,
            trajectory_encoder=self.traj_encoder)
        super(NPOTaskEmbedding, self).__init__(
            sampler_cls=sampler_cls, sampler_args=sampler_args, **kwargs)

    @overrides
    def init_opt(self):
        pol_loss, pol_mean_kl, pol_input_list = self._init_policy_opt()

        task_enc_loss, task_enc_mean_kl, task_enc_input_list = self._init_task_encoder_opt(
        )

        traj_enc_loss, traj_enc_mean_kl, traj_enc_input_list = self._init_traj_encoder_opt(
        )

        #TODO: CE loss function

        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy,
            leq_constraint=(pol_mean_kl, self.step_size),
            inputs=pol_input_list,
            constraint_name="mean_kl")

        self.task_encoder_optimizer.update_opt(
            loss=task_enc_loss,
            target=self.task_encoder,
            leq_constraint=(task_enc_mean_kl, self.task_encoder_step_size),
            inputs=task_enc_input_list,
            constraint_name="task_encoder_mean_kl")

        self.traj_encoder_optimizer.update_opt(
            loss=traj_enc_loss,
            target=self.traj_encoder,
            leq_constraint=(traj_enc_mean_kl, self.traj_encoder_step_size),
            inputs=traj_enc_input_list,
            constraint_name="trajectory_encoder_mean_kl")

        return dict()

    def _init_policy_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = tensor_utils.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [
            old_dist_info_vars[k] for k in dist.dist_info_keys
        ]

        state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [
            state_info_vars[k] for k in self.policy.state_info_keys
        ]

        # TODO CE loss function
        if is_recurrent:
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars,
                                       dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = -tf.reduce_sum(
                lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr * advantage_var)

        input_list = [
            obs_var,
            action_var,
            advantage_var,
        ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        return surr_loss, mean_kl, input_list

    def _init_task_encoder_opt(self):
        is_recurrent = int(self.task_encoder.recurrent)
        task_var = self.env.task_space.new_tensor_variable(
            'task',
            extra_dims=1 + is_recurrent,
        )
        latent_var = self.task_encoder.latent_space.new_tensor_variable(
            'task_enc_latent',
            extra_dims=1 + is_recurrent,
        )
        dist = self.task_encoder.distribution

        old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='task_enc_old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [
            old_dist_info_vars[k] for k in dist.dist_info_keys
        ]

        state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='task_enc_%s' % k)
            for k, shape in self.task_encoder.state_info_specs
        }
        state_info_vars_list = [
            state_info_vars[k] for k in self.task_encoder.state_info_keys
        ]

        # TODO: cross-entropy loss function
        if is_recurrent:
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="task_enc_valid")
        else:
            valid_var = None

        dist_info_vars = self.task_encoder.dist_info_sym(
            task_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(latent_var, old_dist_info_vars,
                                       dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = -tf.reduce_sum(lr * valid_var) / tf.reduce_sum(
                valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr)

        input_list = [
            task_var,
            latent_var,
        ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        return surr_loss, mean_kl, input_list

    def _init_traj_encoder_opt(self):
        is_recurrent = int(self.traj_encoder.recurrent)

        # # TODO: (a, s^H)
        traj_var = self.traj_encoder.input_space.new_tensor_variable(
            'traj_enc_input',
            extra_dims=1 + is_recurrent,
        )
        latent_var = self.traj_encoder.latent_space.new_tensor_variable(
            'traj_enc_latent',
            extra_dims=1 + is_recurrent,
        )
        dist = self.traj_encoder.distribution

        old_dist_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='traj_enc_old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [
            old_dist_info_vars[k] for k in dist.dist_info_keys
        ]

        state_info_vars = {
            k: tf.placeholder(
                tf.float32,
                shape=[None] * (1 + is_recurrent) + list(shape),
                name='traj_enc_%s' % k)
            for k, shape in self.traj_encoder.state_info_specs
        }
        state_info_vars_list = [
            state_info_vars[k] for k in self.traj_encoder.state_info_keys
        ]

        # TODO cross-entropy loss function
        if is_recurrent:
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="traj_enc_valid")
        else:
            valid_var = None

        dist_info_vars = self.traj_encoder.dist_info_sym(
            traj_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(latent_var, old_dist_info_vars,
                                       dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = -tf.reduce_sum(lr * valid_var) / tf.reduce_sum(
                valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr)

        input_list = [
            traj_var,
            latent_var,
        ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        return surr_loss, mean_kl, input_list

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(
            ext.extract(samples_data, "observations", "actions", "advantages"))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [
            agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"], )
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

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
