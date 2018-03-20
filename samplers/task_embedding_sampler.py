import numpy as np

import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler

from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.spaces.box import Box

from sandbox.embed2learn.embeddings.utils import concat_spaces

from rllab.algos import util  # DEBUG
from rllab.misc import special  # DEBUG

singleton_pool = parallel_sampler.singleton_pool


def rollout(env,
            agent,
            task_encoder,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):
    env_observations = []
    observations = []
    tasks = []
    latents = []
    latent_infos = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    o = env.reset()
    agent.reset()

    # Append latent vector to observation
    # TODO: should we sample every step or every rollout?
    obs_embed_space = concat_spaces(env.observation_space,
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
        latents.append(task_encoder.latent_space.flatten(z))
        latent_infos.append(latent_info)
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
        latents=tensor_utils.stack_tensor_list(latents),
        latent_infos=tensor_utils.stack_tensor_list(latent_infos),
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
        path = rollout(G.env, G.policy, G.task_encoder, max_path_length)
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
        baselines = []
        returns = []
        trajectories = []

        max_path_length = self.algo.max_path_length
        action_space = self.algo.env.action_space
        observation_space = self.algo.env.observation_space

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.algo.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["deltas"] = deltas

        # calculate trajectory tensors (TODO: probably can do this in TF)
        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path["returns"] = special.discount_cumsum(path["rewards"],
                                                      self.algo.discount)
            returns.append(path["returns"])

            # trajectories
            act = path['actions']
            obs = path['observations']
            act_flat = action_space.flatten_n(act)
            obs_flat = observation_space.flatten_n(obs)
            traj = np.concatenate([act_flat, obs_flat], axis=1)
            traj = np.concatenate(traj)
            trajs = np.tile(traj, (max_path_length, 1))
            path['trajectories'] = trajs
            trajectories.append(path['trajectories'])

        ev = special.explained_variance_1d(
            np.concatenate(baselines), np.concatenate(returns))

        #DEBUG CPU vars ######################
        cpu_adv = tensor_utils.concat_tensor_list(
            [path["advantages"] for path in paths])
        cpu_deltas = tensor_utils.concat_tensor_list(
            [path["deltas"] for path in paths])
        cpu_act = tensor_utils.concat_tensor_list(
            [path["actions"] for path in paths])
        cpu_obs = tensor_utils.concat_tensor_list(
            [path["observations"] for path in paths])
        cpu_agent_infos = tensor_utils.concat_tensor_dict_list(
            [path["agent_infos"] for path in paths])

        if self.algo.center_adv:
            cpu_adv = util.center_advantages(cpu_adv)

        if self.algo.positive_adv:
            cpu_adv = util.shift_advantages_to_positive(cpu_adv)
        #####################################

        # make all paths the same length
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        latents = [path['latents'] for path in paths]
        latents = tensor_utils.pad_tensor_n(latents, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        #trajectories = tensor_utils.pad_tensor_n(trajectories, max_path_length)

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.sum(
            self.algo.policy.distribution.entropy(agent_infos) * valids
        ) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            latents=latents,
            trajectories=trajectories,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
            cpu_adv=cpu_adv,  #DEBUG
            cpu_deltas=cpu_deltas,  #DEBUG
            cpu_obs=cpu_obs,  #DEBUG
            cpu_act=cpu_act,  #DEBUG
            cpu_agent_infos=cpu_agent_infos,  # DEBUG
        )

        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

    #TODO: embedding-specific diagnostics
    def log_diagnostics(self, paths):
        return super(TaskEmbeddingSampler, self).log_diagnostics(paths)