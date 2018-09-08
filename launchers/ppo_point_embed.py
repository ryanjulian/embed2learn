from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from garage.baselines import LinearFeatureBaseline
from garage.envs.env_spec import EnvSpec
from garage.misc.instrument import run_experiment
from garage.tf.spaces import Box

from sandbox.embed2learn.algos import PPOTaskEmbedding
from sandbox.embed2learn.baselines import MultiTaskLinearFeatureBaseline
from sandbox.embed2learn.baselines import MultiTaskGaussianMLPBaseline
from sandbox.embed2learn.envs import PointEnv
from sandbox.embed2learn.envs import MultiTaskEnv
from sandbox.embed2learn.envs.multi_task_env import TfEnv
from sandbox.embed2learn.embeddings import EmbeddingSpec
from sandbox.embed2learn.embeddings import GaussianMLPEmbedding
from sandbox.embed2learn.embeddings.utils import concat_spaces
from sandbox.embed2learn.policies import GaussianMLPMultitaskPolicy


def circle(r, n):
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)


N = 4
goals = circle(3.0, N)
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g,
            'completion_bonus': 0.,
            'action_scale': 0.1,
            'random_start': False,
            'never_done': True,
            'obs_noise': 0.,
        }
    }
    for i, g in enumerate(goals)
}


def run_task(v):
    v = SimpleNamespace(**v)

    task_names = sorted(v.tasks.keys())
    task_args = [v.tasks[t]['args'] for t in task_names]
    task_kwargs = [v.tasks[t]['kwargs'] for t in task_names]

    # Environment
    env = TfEnv(
        MultiTaskEnv(
            task_env_cls=PointEnv,
            task_args=task_args,
            task_kwargs=task_kwargs))

    # Latent space and embedding specs
    # TODO(gh/10): this should probably be done in Embedding or Algo
    latent_lb = np.zeros(v.latent_length, )
    latent_ub = np.ones(v.latent_length, )
    latent_space = Box(latent_lb, latent_ub)

    # trajectory space is (TRAJ_ENC_WINDOW, act_obs) where act_obs is a stacked
    # vector of flattened actions and observations
    act_lb, act_ub = env.action_space.bounds
    act_lb_flat = env.action_space.flatten(act_lb)
    act_ub_flat = env.action_space.flatten(act_ub)
    obs_lb, obs_ub = env.observation_space.bounds
    obs_lb_flat = env.observation_space.flatten(obs_lb)
    obs_ub_flat = env.observation_space.flatten(obs_ub)
    # act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
    # act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
    act_obs_lb = obs_lb_flat
    act_obs_ub = obs_ub_flat
    # act_obs_lb = act_lb_flat
    # act_obs_ub = act_ub_flat
    traj_lb = np.stack([act_obs_lb] * v.inference_window)
    traj_ub = np.stack([act_obs_ub] * v.inference_window)
    traj_space = Box(traj_lb, traj_ub)

    embedding_spec = EmbeddingSpec(env.task_space, latent_space)
    inference_spec = EmbeddingSpec(traj_space, latent_space)
    task_obs_space = concat_spaces(env.task_space, env.observation_space)
    env_spec_embed = EnvSpec(task_obs_space, env.action_space)

    inference = GaussianMLPEmbedding(
        name="inference",
        embedding_spec=inference_spec,
        hidden_sizes=(64, 64),
        std_share_network=True,
        init_std=1.0,
    )

    # Embeddings
    embedding = GaussianMLPEmbedding(
        name="embedding",
        embedding_spec=embedding_spec,
        hidden_sizes=(32, 32),
        std_share_network=True,
        hidden_nonlinearity=tf.nn.relu,
        init_std=1.0,
    )

    # Multitask policy
    policy = GaussianMLPMultitaskPolicy(
        name="policy",
        env_spec=env.spec,
        task_space=env.task_space,
        embedding=embedding,
        hidden_sizes=(32, 16),
        std_share_network=True,
        init_std=v.policy_init_std,
        stop_latent_gradient=True,
    )

    extra = v.latent_length + len(v.tasks)  # + traj_space.flat_dim
    baseline = MultiTaskGaussianMLPBaseline(
        env_spec=env.spec, extra_dims=extra)

    algo = PPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        inference=inference,
        batch_size=v.batch_size,
        max_path_length=v.max_path_length,
        n_itr=500,
        discount=0.99,
        step_size=0.2,
        plot=True,
        stop_policy_ent_gradient=True,
        policy_ent_coeff=v.policy_ent_coeff,
        embedding_ent_coeff=v.embedding_ent_coeff,
        inference_ce_coeff=v.inference_ce_coeff,
        inference_reg_coeff=v.inference_reg_coeff,
    )
    algo.train()


config = dict(
    tasks=TASKS,
    latent_length=2,
    inference_window=2,
    batch_size=4096 * len(TASKS),
    policy_ent_coeff=1e-3,
    embedding_ent_coeff=1e-3,
    embedding_reg_coeff=1e-1,
    inference_ce_coeff=1e-3,
    inference_reg_coeff=0.,
    max_path_length=50,
    policy_init_std=2.0,
)

run_experiment(
    run_task,
    exp_prefix='ppo_point_embed',
    n_parallel=16,
    seed=1,
    variant=config,
    plot=True,
)
