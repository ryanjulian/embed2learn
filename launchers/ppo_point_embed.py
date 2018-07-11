import numpy as np

from garage.baselines import LinearFeatureBaseline
from garage.envs.env_spec import EnvSpec
from garage.misc.ext import set_seed
from garage.misc.instrument import run_experiment
from garage.tf.spaces import Box

from sandbox.embed2learn.algos import PPOTaskEmbedding
from sandbox.embed2learn.algos.trpo_task_embedding import KLConstraint
from sandbox.embed2learn.baselines import MultiTaskLinearFeatureBaseline
from sandbox.embed2learn.embeddings import GaussianMLPEmbedding
from sandbox.embed2learn.embeddings import GaussianMLPMultitaskPolicy
from sandbox.embed2learn.embeddings import EmbeddingSpec
from sandbox.embed2learn.envs import PointEnv
from sandbox.embed2learn.envs import MultiTaskEnv
from sandbox.embed2learn.envs.multi_task_env import TfEnv
from sandbox.embed2learn.envs.multi_task_env import normalize
from sandbox.embed2learn.embeddings.utils import concat_spaces


def circle(r, n):
    for t in np.arange(0, 2 * np.pi, 2 * np.pi / n):
        yield r * np.sin(t), r * np.cos(t)


goals = circle(3.0, 12)
TASKS = {
    str(i + 1): {
        'args': [],
        'kwargs': {
            'goal': g
        }
    }
    for i, g in enumerate(goals)
}


TASKS = {
    '(-3, 0)': {'args': [], 'kwargs': {'goal': (-3, 0)}},
    '(3, 0)': {'args': [], 'kwargs': {'goal': (3, 0)}},
    '(0, 3)': {'args': [], 'kwargs': {'goal': (0, 3)}},
    '(0, -0)': {'args': [], 'kwargs': {'goal': (0, -3)}},
}  # yapf: disable
TASK_NAMES = sorted(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]

# Embedding params
LATENT_LENGTH = 2
TRAJ_ENC_WINDOW = 1


def run_task(*_):
    set_seed(1)

    # Environment
    env = TfEnv(
        normalize(
            MultiTaskEnv(
                task_env_cls=PointEnv,
                task_args=TASK_ARGS,
                task_kwargs=TASK_KWARGS)))

    # Latent space and embedding specs
    # TODO(gh/10): this should probably be done in Embedding or Algo
    latent_lb = np.zeros(LATENT_LENGTH, )
    latent_ub = np.ones(LATENT_LENGTH, )
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
    traj_lb = np.stack([act_obs_lb] * TRAJ_ENC_WINDOW)
    traj_ub = np.stack([act_obs_ub] * TRAJ_ENC_WINDOW)
    traj_space = Box(traj_lb, traj_ub)

    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
    task_obs_space = concat_spaces(env.task_space, env.observation_space)
    env_spec_embed = EnvSpec(task_obs_space, env.action_space)

    # TODO(): rename to inference_network
    traj_embedding = GaussianMLPEmbedding(
        name="inference",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(20, 10),  # was the same size as policy in Karol's paper
        std_share_network=True,
        init_std=1.0,
    )

    # Embeddings
    task_embedding = GaussianMLPEmbedding(
        name="embedding",
        embedding_spec=task_embed_spec,
        hidden_sizes=(20, 20),
        std_share_network=True,
        init_std=6.0,
        max_std=8.0,
    )

    # Multitask policy
    policy = GaussianMLPMultitaskPolicy(
        name="policy",
        env_spec=env.spec,
        task_space=env.task_space,
        embedding=task_embedding,
        hidden_sizes=(20, 10),
        std_share_network=True,
        init_std=8.0,
        max_std=16.0,
        min_std=6.0,
    )

    baseline = MultiTaskLinearFeatureBaseline(env_spec=env_spec_embed)

    temp = 1.0
    algo = PPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        inference=traj_embedding,
        batch_size=4096,  # 4096
        max_path_length=50,
        n_itr=1000,
        discount=0.99,
        step_size=0.2,
        plot=True,
        policy_ent_coeff=temp * 50e-3,  # 5e-2
        embedding_ent_coeff=temp * 3e-3,  # 1e-3
        inference_ce_coeff=temp * 200e-3,  # 2e-1
    )
    algo.train()


run_experiment(
    run_task,
    exp_prefix='ppo_point_embed',
    n_parallel=16,
    plot=False,
)

# run_task()
