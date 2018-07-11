import pickle
import time

import numpy as np

from garage.sampler import utils  # DEBUG
from garage.misc import ext
from garage.misc import special  # DEBUG
import garage.misc.logger as logger
from garage.sampler import parallel_sampler
from garage.sampler.stateful_pool import singleton_pool

from garage.tf.misc import tensor_utils
from garage.tf.samplers.batch_sampler import BatchSampler
from garage.tf.samplers.batch_sampler import worker_init_tf
from garage.tf.samplers.batch_sampler import worker_init_tf_vars
from garage.tf.spaces import Box

from sandbox.embed2learn.policies import MultitaskPolicy
from sandbox.embed2learn.embeddings.utils import concat_spaces
from sandbox.embed2learn.envs import MultiTaskEnv
from sandbox.embed2learn.samplers.utils import sliding_window

# TODO: improvements to garage so that you don't need to rwrite a whole sampler
# to change the rollout process


def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):

    observations = []
    tasks = []
    latents = []
    latent_infos = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    # Resets
    o = env.reset()
    agent.reset()

    # Sample embedding network
    # NOTE: it is important to do this _once per rollout_, not once per
    # timestep, since we need correlated noise.
    t = env.active_task_one_hot
    z, latent_info = agent.get_latent(t)

    if animated:
        env.render()

    path_length = 0
    while path_length < max_path_length:
        #a, agent_info = agent.get_action(np.concatenate((t, o)))
        a, agent_info = agent.get_action_from_latent(z, o)
        # latent_info = agent_info["latent_info"]
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        tasks.append(t)
        # z = latent_info["mean"]
        latents.append(agent.latent_space.flatten(z))
        latent_infos.append(latent_info)
        rewards.append(r)
        actions.append(agent.action_space.flatten(a))
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
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        tasks=tensor_utils.stack_tensor_list(tasks),
        latents=tensor_utils.stack_tensor_list(latents),
        latent_infos=tensor_utils.stack_tensor_dict_list(latent_infos),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


# parallel_sampler worker API
def _worker_populate_task(g, env, policy, inference, scope=None):
    g = parallel_sampler._get_scoped_g(g, scope)
    g.env = pickle.loads(env)
    g.policy = pickle.loads(policy)


def _worker_terminate_task(g, scope=None):
    g = parallel_sampler._get_scoped_g(g, scope)
    if getattr(g, "env", None):
        g.env.close()
        g.env = None
    if getattr(g, "policy", None):
        g.policy.terminate()
        g.policy = None
    if getattr(g, "inference", None):
        g.inference.terminate()
        g.inference = None


def _worker_set_inference_params(g, params, scope=None):
    g = parallel_sampler._get_scoped_g(g, scope)
    # g.inference.set_param_values(params)


def _worker_collect_one_path(g, max_path_length, scope=None):
    g = parallel_sampler._get_scoped_g(g, scope)
    path = rollout(g.env, g.policy, max_path_length)
    return path, len(path["rewards"])


#TODO: can this use VectorizedSampler?
class TaskEmbeddingSampler(BatchSampler):
    def __init__(self, *args, inference=None, **kwargs):
        super(TaskEmbeddingSampler, self).__init__(*args, **kwargs)
        self.inference = inference

    def populate_task(self, env, policy, scope=None):
        logger.log("Populating workers...")
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(_worker_populate_task, [
                (pickle.dumps(env), pickle.dumps(policy), scope)
            ] * singleton_pool.n_parallel)
        else:
            # avoid unnecessary copying
            g = parallel_sampler._get_scoped_g(singleton_pool.G, scope)
            g.env = env
            g.policy = policy
        parallel_sampler.set_seed(ext.get_seed())
        logger.log("Populated")

    def terminate_task(self, scope=None):
        singleton_pool.run_each(_worker_terminate_task,
                                [(scope, )] * singleton_pool.n_parallel)

    # BatchSampler API
    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        self.populate_task(self.algo.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        self.terminate_task(scope=self.algo.scope)

    def sample_paths(self,
                     policy_params,
                     max_samples,
                     max_path_length,
                     env_params=None,
                     inference_params=None,
                     scope=None):
        singleton_pool.run_each(
            parallel_sampler._worker_set_policy_params,
            [(policy_params, scope)] * singleton_pool.n_parallel,
        )
        singleton_pool.run_each(
            _worker_set_inference_params,
            [(inference_params, scope)] * singleton_pool.n_parallel,
        )
        if env_params:
            singleton_pool.run_each(
                parallel_sampler._worker_set_env_params,
                [(env_params, scope)] * singleton_pool.n_parallel,
            )

        return singleton_pool.run_collect(
            _worker_collect_one_path,
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

    #TODO: vectorize
    def process_samples(self, itr, paths):
        baselines = []
        returns = []

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

            # Calculate trajectory samples
            #
            # Pad and flatten action and observation traces
            act = tensor_utils.pad_tensor(path['actions'], max_path_length)
            obs = tensor_utils.pad_tensor(path['observations'],
                                          max_path_length)
            act_flat = action_space.flatten_n(act)
            obs_flat = observation_space.flatten_n(obs)
            # Create a time series of stacked [act, obs] vectors
            #XXX now the inference network only looks at obs vectors
            #act_obs = np.concatenate([act_flat, obs_flat], axis=1)  # TODO reactivate for harder envs?
            act_obs = obs_flat
            # act_obs = act_flat
            # Calculate a forward-looking sliding window of the stacked vectors
            #
            # If act_obs has shape (n, d), then trajs will have shape
            # (n, window, d)
            #
            # The length of the sliding window is determined by the trajectory
            # inference spec. We smear the last few elements to preserve the
            # time dimension.
            window = self.inference.input_space.shape[0]
            trajs = sliding_window(act_obs, window, 1, smear=True)
            trajs_flat = self.inference.input_space.flatten_n(trajs)
            path['trajectories'] = trajs_flat

            # trajectory infos
            _, traj_infos = self.inference.get_latents(trajs)
            path['trajectory_infos'] = traj_infos

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
            cpu_adv = utils.center_advantages(cpu_adv)

        if self.algo.positive_adv:
            cpu_adv = utils.shift_advantages_to_positive(cpu_adv)
        #####################################

        # make all paths the same length
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        tasks = [path["tasks"] for path in paths]
        tasks = tensor_utils.pad_tensor_n(tasks, max_path_length)

        latents = [path['latents'] for path in paths]
        latents = tensor_utils.pad_tensor_n(latents, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        trajectories = tensor_utils.stack_tensor_list(
            [path["trajectories"] for path in paths])

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        latent_infos = [path["latent_infos"] for path in paths]
        latent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in latent_infos
        ])

        trajectory_infos = [path["trajectory_infos"] for path in paths]
        trajectory_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in trajectory_infos
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
            self.algo.policy.distribution.entropy(agent_infos) *
            valids) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            tasks=tasks,
            latents=latents,
            trajectories=trajectories,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            latent_infos=latent_infos,
            trajectory_infos=trajectory_infos,
            env_infos=env_infos,
            paths=paths,
            cpu_adv=cpu_adv,  #DEBUG
            cpu_deltas=cpu_deltas,  #DEBUG
            cpu_obs=cpu_obs,  #DEBUG
            cpu_act=cpu_act,  #DEBUG
            cpu_agent_infos=cpu_agent_infos,  # DEBUG
        )

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
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
