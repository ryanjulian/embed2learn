import pickle
import time

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.misc import tensor_utils
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
from rllab.sampler.utils import rollout

from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.spaces.box import Box

from sandbox.embed2learn.envs.multi_task_env import MultiTaskEnv
from sandbox.embed2learn.embeddings.base import Embedding

singleton_pool = parallel_sampler.singleton_pool


def _optimizer_or_default(optimizer, args):
    use_optimizer = optimizer
    use_args = args
    if use_optimizer is None:
        if use_args is None:
            use_args = dict(name="optimizer")
        use_optimizer = PenaltyLbfgsOptimizer(**use_args)
    return use_optimizer


def _obs_embed_space(observation_space, latent_space):
    obs_lb, obs_ub = observation_space.bounds
    latent_lb, latent_ub = latent_space.bounds
    return Box(
        np.concatenate([latent_lb, obs_lb]),
        np.concatenate([latent_ub, obs_ub]))


def _rollout(env,
             agent,
             task_encoder,
             max_path_length=np.inf,
             animated=False,
             speedup=1,
             always_return_paths=False):
    env_observations = []
    observations = []
    tasks = []
    task_latents = []
    task_latent_infos = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    o = env.reset()
    agent.reset()

    # Append latent vector to observation
    # TODO: should we sample every step or every rollout?
    obs_embed_space = _obs_embed_space(env.observation_space,
                                       task_encoder.latent_space)

    if animated:
        env.render()

    path_length = 0
    while path_length < max_path_length:
        t = env.active_task_one_hot
        z, latent_info = task_encoder.get_latent(t)
        z_o = np.concatenate([z, o])
        a, agent_info = agent.get_action(z_o)
        next_o, r, d, env_info = env.step(a)
        env_observations.append(env.observation_space.flatten(o))
        observations.append(obs_embed_space.flatten(z_o))
        tasks.append(t)
        task_latents.append(task_encoder.latent_space.flatten(z))
        task_latent_infos.append(latent_info)
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        env_observations=tensor_utils.stack_tensor_list(env_observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        tasks=tensor_utils.stack_tensor_list(tasks),
        task_latents=tensor_utils.stack_tensor_list(task_latents),
        task_latent_infos=tensor_utils.stack_tensor_list(task_latent_infos),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


#TODO: can this use VectorizedSampler?
class TaskEmbeddingSampler(BatchSampler):
    def __init__(self,
                 *args,
                 task_encoder=None,
                 trajectory_encoder=None,
                 trajectory_encoder_ent_coeff=None,
                 policy_ent_coeff=None,
                 **kwargs):
        super(TaskEmbeddingSampler, self).__init__(*args, **kwargs)
        self.task_encoder = task_encoder
        self.traj_encoder = trajectory_encoder
        self.traj_encoder_ent_coeff = trajectory_encoder_ent_coeff
        self.pol_ent_coeff = policy_ent_coeff

    # parallel_sampler API
    # TODO: figure out how to avoid copying all this code
    def _worker_populate_task(self, G, env, policy, task_encoder, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        G.env = pickle.loads(env)
        G.policy = pickle.loads(policy)
        G.task_encoder = pickle.loads(task_encoder)

    def _worker_terminate_task(self, G, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        if getattr(G, "env", None):
            G.env.terminate()
            G.env = None
        if getattr(G, "policy", None):
            G.policy.terminate()
            G.policy = None
        if getattr(G, "task_encoder", None):
            G.task_encoder.terminate()
            G.task_encoder = None

    def _worker_set_task_encoder_params(self, G, params, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        G.task_encoder.set_param_values(params)

    def _worker_collect_one_path(self, G, max_path_length, scope=None):
        G = parallel_sampler._get_scoped_G(G, scope)
        path = _rollout(G.env, G.policy, G.task_encoder, max_path_length)
        return path, len(path["rewards"])

    def populate_task(self, env, policy, task_encoder, scope=None):
        logger.log("Populating workers...")
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(self._worker_populate_task,
                                    [(pickle.dumps(env), pickle.dumps(policy),
                                      pickle.dumps(task_encoder),
                                      scope)] * singleton_pool.n_parallel)
        else:
            # avoid unnecessary copying
            G = parallel_sampler._get_scoped_G(singleton_pool.G, scope)
            G.env = env
            G.policy = policy
            G.task_encoder = task_encoder
        logger.log("Populated")

    def terminate_task(self, scope=None):
        singleton_pool.run_each(self._worker_terminate_task,
                                [(scope, )] * singleton_pool.n_parallel)

    # BatchSampler API
    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(BatchSampler.worker_init_tf)
        self.populate_task(self.algo.env, self.algo.policy,
                           self.algo.task_encoder)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(BatchSampler.worker_init_tf_vars)

    def shutdown_worker(self):
        self.terminate_task(scope=self.algo.scope)

    def sample_paths(self,
                     policy_params,
                     max_samples,
                     max_path_length,
                     env_params=None,
                     task_encoder_params=None,
                     scope=None):
        singleton_pool.run_each(
            parallel_sampler._worker_set_policy_params,
            [(policy_params, scope)] * singleton_pool.n_parallel,
        )
        singleton_pool.run_each(
            self._worker_set_task_encoder_params,
            [(task_encoder_params, scope)] * singleton_pool.n_parallel,
        )
        if env_params:
            singleton_pool.run_each(
                parallel_sampler._worker_set_env_params,
                [(env_params, scope)] * singleton_pool.n_parallel,
            )

        return singleton_pool.run_collect(
            self._worker_collect_one_path,
            threshold=max_samples,
            args=(max_path_length, scope),
            show_prog_bar=True,
        )

    def obtain_samples(self, itr):
        policy_params = self.algo.policy.get_param_values()
        env_params = self.algo.env.get_param_values()
        task_enc_params = self.algo.task_encoder.get_param_values()
        paths = self.sample_paths(
            policy_params=policy_params,
            env_params=env_params,
            task_encoder_params=task_enc_params,
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

    #TODO: vectorize
    def process_samples(self, itr, paths):
        # Augment the rewards with task encoder and policy entropies
        action_space = self.algo.env.action_space
        observation_space = self.algo.env.observation_space
        max_path_length = self.algo.max_path_length
        for idx, path in enumerate(paths):
            rewards_augmented = path['rewards']

            # Calculate latent log likelihoods given the trajectory
            ## Pad paths to ensure the trajectory encoder has fixed-sized input
            act = tensor_utils.pad_tensor(path['actions'], max_path_length)
            obs = tensor_utils.pad_tensor(path['env_observations'],
                                          max_path_length)
            latents = tensor_utils.pad_tensor(path['task_latents'],
                                              max_path_length)
            act_flat = action_space.flatten_n(act)
            obs_flat = observation_space.flatten_n(obs)
            traj = np.concatenate([act_flat, obs_flat], axis=1)
            traj = np.concatenate(traj)
            trajs = np.tile(traj, (max_path_length, 1))
            latent_loglis = self.traj_encoder.log_likelihoods(trajs, latents)
            rewards_augmented += self.traj_encoder_ent_coeff * \
                latent_loglis[:len(rewards_augmented)] # disregards padding

            # Calculate policy entropies
            pol_dist = self.algo.policy.distribution
            pol_ents = pol_dist.entropy(path['agent_infos'])
            rewards_augmented += self.pol_ent_coeff * pol_ents

            # Replace rewards with augmented rewards
            path['rewards'] = rewards_augmented
            path['trajectories'] = trajs

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
                 policy_ent_coeff=0.01,
                 task_encoder=None,
                 task_encoder_optimizer=None,
                 task_encoder_optimizer_args=None,
                 task_encoder_step_size=0.01,
                 task_encoder_ent_coeff=0.01,
                 trajectory_encoder=None,
                 trajectory_encoder_optimizer=None,
                 trajectory_encoder_optimizer_args=None,
                 trajectory_encoder_step_size=0.01,
                 trajectory_encoder_ent_coeff=0.01,
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
        self.policy_ent_coeff = policy_ent_coeff

        # Task encoder optimizer
        self.task_encoder_optimizer = _optimizer_or_default(
            task_encoder_optimizer, task_encoder_optimizer_args)
        self.task_encoder_step_size = task_encoder_step_size
        self.task_encoder_ent_coeff = task_encoder_ent_coeff

        # Trajectory encoder optimizer
        self.traj_encoder_optimizer = _optimizer_or_default(
            trajectory_encoder_optimizer, trajectory_encoder_optimizer_args)
        self.traj_encoder_step_size = trajectory_encoder_step_size
        self.traj_encoder_ent_coeff = trajectory_encoder_ent_coeff

        sampler_cls = TaskEmbeddingSampler
        sampler_args = dict(
            task_encoder=self.task_encoder,
            trajectory_encoder=self.traj_encoder,
            trajectory_encoder_ent_coeff=self.traj_encoder_ent_coeff,
            policy_ent_coeff=self.policy_ent_coeff,
        )
        super(NPOTaskEmbedding, self).__init__(
            sampler_cls=sampler_cls, sampler_args=sampler_args, **kwargs)

    @overrides
    def init_opt(self):
        pol_loss, pol_mean_kl, pol_input_list = self._init_policy_opt()

        # task_enc_loss, task_enc_mean_kl, task_enc_input_list = \
        #     self._init_task_encoder_opt()

        # traj_enc_loss, traj_enc_mean_kl, traj_enc_input_list = \
        #     self._init_traj_encoder_opt()

        self.optimizer.update_opt(
            loss=pol_loss,
            target=self.policy,
            leq_constraint=(pol_mean_kl, self.step_size),
            inputs=pol_input_list,
            constraint_name="mean_kl")

        # self.task_encoder_optimizer.update_opt(
        #     loss=task_enc_loss,
        #     target=self.task_encoder,
        #     leq_constraint=(task_enc_mean_kl, self.task_encoder_step_size),
        #     inputs=task_enc_input_list,
        #     constraint_name="task_encoder_mean_kl")

        # self.traj_encoder_optimizer.update_opt(
        #     loss=traj_enc_loss,
        #     target=self.traj_encoder,
        #     leq_constraint=(traj_enc_mean_kl, self.traj_encoder_step_size),
        #     inputs=traj_enc_input_list,
        #     constraint_name="trajectory_encoder_mean_kl")

        return dict()

    def _init_policy_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_embed_space = _obs_embed_space(self.env.observation_space,
                                           self.task_encoder.latent_space)
        obs_var = obs_embed_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        # advantage_var = tensor_utils.new_tensor(
        #     'advantage',
        #     ndim=1 + is_recurrent,
        #     dtype=tf.float32,
        # )
        reward_var = tensor_utils.new_tensor(
            'reward',
            ndim=1 + is_recurrent,
            dtype=tf.float32,
        )
        baseline_var = tensor_utils.new_tensor(
            'baseline',
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

        if is_recurrent:
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        # TODO: add entropy terms
        rewards = reward_var #for now, TODO: add entropy terms

        # Calculate advantages
        pad = tf.constant([[0,0],[0,1]])
        baseline_pad = tf.pad(baseline_var, pad)
        gamma_lambda = tf.constant(self.algo.discount * self.algo.gae_lambda,
            dtype=tf.float32, shape=reward_var.shape)
        discounts = gamma_lambda.cumprod(gamma_lambda, exclusive=True)
        deltas = rewards + \
                 (self.algo.discount * baseline_pad[1:]) + 
                 baseline_pad[:-1]
        advantages = tf.cumsum(discounts * deltas)
        if not self.algo.policy.recurrent:
            # TODO: center advantages
            pass

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars,
                                       dist_info_vars)
        if is_recurrent:
            raise NotImplementedError
            # mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            # surr_loss = -tf.reduce_sum(
            #     lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr * advantages)

        input_list = [
            obs_var,
            action_var,
            reward_var,
            baseline_var,
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
            'task_latent',
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

        if is_recurrent:
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="task_enc_valid")
        else:
            valid_var = None

        dist_info_vars = self.task_encoder.dist_info_sym(
            task_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)

        input_list = [
            task_var,
            latent_var,
        ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        return mean_kl, input_list

    def _init_traj_encoder_opt(self):
        is_recurrent = int(self.traj_encoder.recurrent)

        traj_var = self.traj_encoder.input_space.new_tensor_variable(
            'traj',
            extra_dims=1 + is_recurrent,
        )
        latent_var = self.traj_encoder.latent_space.new_tensor_variable(
            'traj_latent',
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

        if is_recurrent:
            valid_var = tf.placeholder(
                tf.float32, shape=[None, None], name="traj_enc_valid")
        else:
            valid_var = None

        # Loss function
        dist_info_vars = self.traj_encoder.dist_info_sym(
            traj_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        logli = dist.log_likelihood_sym(latent_var, dist_info_vars)
        if is_recurrent:
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            surr_loss = -tf.reduce_sum(
                lr * advantage_var * valid_var) / tf.reduce_sum(valid_var)
        else:
            mean_kl = tf.reduce_mean(kl)
            surr_loss = -tf.reduce_mean(lr * advantage_var)

        input_list = [
            traj_var,
            latent_var,
        ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        return mean_kl, input_list

    @overrides
    def optimize_policy(self, itr, samples_data):
        # Policy
        logger.log("### Policy ###")

        # Calculate baselines
        # note: fitting step 
        if hasattr(self.algo.baseline, 'predict_n'):
            baselines = self.algo.baseline.predict_n(paths)
        else:
            baselines = [self.algo.baseline.predict(path) for path in paths]

        # all_input_values = tuple(
        #     ext.extract(samples_data, "observations", "actions", "advantages"))
        all_input_values = tuple(
            ext.extract(samples_data, 'observations', 'actions', 'rewards'))

        # add policy params
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [
            agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        # add valids for a recurrent policy
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

        # # Task encoder
        # logger.log("### Task Encoder ###")
        # tasks = tensor_utils.concat_tensor_list(
        #     [path['tasks'] for path in paths])
        # latents = tensor_utils.concat_tensor_list(
        #     [path['task_latents'] for path in paths])
        # all_input_values = tuple(tasks, latents)
        # latent_infos = tensor_utils.concat_tensor_list(
        #     [path['task_latent_infos'] for path in paths])
        # state_info_list = [
        #     latent_infos[k] for k in self.task_encoder.state_info_keys
        # ]
        # dist_info_list = [
        #     agent_infos[k]
        #     for k in self.task_encoder.distribution.dist_info_keys
        # ]
        # all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        # if self.task_encoder.recurrent:
        #     all_input_values += (samples_data["valids"], )

        # logger.log("Computing loss before")
        # loss_before = self.task_encoder_optimizer.loss(all_input_values)
        # logger.log("Computing KL before")
        # mean_kl_before = self.task_encoder_optimizer.constraint_val(
        #     all_input_values)
        # logger.log("Optimizing")
        # self.task_encoder_optimizer.optimize(all_input_values)
        # logger.log("Computing KL after")
        # mean_kl = self.task_encoder_optimizer.constraint_val(all_input_values)
        # logger.log("Computing loss after")
        # loss_after = self.task_encoder_optimizer.loss(all_input_values)
        # logger.record_tabular('LossBefore', loss_before)
        # logger.record_tabular('LossAfter', loss_after)
        # logger.record_tabular('MeanKLBefore', mean_kl_before)
        # logger.record_tabular('MeanKL', mean_kl)
        # logger.record_tabular('dLoss', loss_before - loss_after)

        # # Trajectory encoder
        # logger.log("### Trajectory Encoder ###")
        # trajs = tensor_utils.concat_tensor_list(
        #     [path['trajectories'] for path in paths])
        # all_input_values = tuple(trajs, latents)
        # agent_infos = samples_data["agent_infos"]
        # state_info_list = [
        #     agent_infos[k] for k in self.traj_encoder.state_info_keys
        # ]
        # dist_info_list = [
        #     agent_infos[k]
        #     for k in self.traj_encoder.distribution.dist_info_keys
        # ]
        # all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        # if self.task_encoder.recurrent:
        #     all_input_values += (samples_data["valids"], )

        # logger.log("Computing loss before")
        # loss_before = self.traj_encoder_optimizer.loss(all_input_values)
        # logger.log("Computing KL before")
        # mean_kl_before = self.traj_encoder_optimizer.constraint_val(
        #     all_input_values)
        # logger.log("Optimizing")
        # self.traj_encoder_optimizer.optimize(all_input_values)
        # logger.log("Computing KL after")
        # mean_kl = self.traj_encoder_optimizer.constraint_val(all_input_values)
        # logger.log("Computing loss after")
        # loss_after = self.traj_encoder_optimizer.loss(all_input_values)
        # logger.record_tabular('LossBefore', loss_before)
        # logger.record_tabular('LossAfter', loss_after)
        # logger.record_tabular('MeanKLBefore', mean_kl_before)
        # logger.record_tabular('MeanKL', mean_kl)
        # logger.record_tabular('dLoss', loss_before - loss_after)

        return dict()

    def train(self, sess=None):
        created_session = True if (sess is None) else False
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
                logger.log("Optimizing policy...")
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
                    _rollout(
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
            trajectory_encoder=self.trajectory_encoder,
        )
