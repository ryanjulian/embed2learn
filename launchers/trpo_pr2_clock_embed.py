from akro.tf import Box
from garage.baselines import LinearFeatureBaseline
from garage.envs import EnvSpec
from garage.misc.instrument import run_experiment
from garage.tf.policies import GaussianMLPPolicy
import numpy as np

from embed2learn.algos import TRPOTaskEmbedding
from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.envs.mujoco import PR2ArmClockEnv
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.envs.multi_task_env import normalize
from embed2learn.embeddings.utils import concat_spaces


N_PARALLEL = 1  # TODO(gh/10): the sampler is broken for n_parallel > 1

TASKS = {
    'center': {'args': [], 'kwargs': {'target': 'center'}},
    'hour_12': {'args': [], 'kwargs': {'target': 'hour_12'}},
    # 'hour_1': {'args': [], 'kwargs': {'target': 'hour_1'}},
    # 'hour_2': {'args': [], 'kwargs': {'target': 'hour_2'}},
    # 'hour_3': {'args': [], 'kwargs': {'target': 'hour_3'}},
    # 'hour_4': {'args': [], 'kwargs': {'target': 'hour_4'}},
    # 'hour_5': {'args': [], 'kwargs': {'target': 'hour_5'}},
    # 'hour_6': {'args': [], 'kwargs': {'target': 'hour_6'}},
    # 'hour_7': {'args': [], 'kwargs': {'target': 'hour_7'}},
    # 'hour_8': {'args': [], 'kwargs': {'target': 'hour_8'}},
    # 'hour_9': {'args': [], 'kwargs': {'target': 'hour_9'}},
    # 'hour_10': {'args': [], 'kwargs': {'target': 'hour_10'}},
    # 'hour_11': {'args': [], 'kwargs': {'target': 'hour_11'}},
} # yapf: disable
TASK_NAMES = sorted(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]

# Embedding parameters
LATENT_LENGTH = 4
TRAJ_ENC_WINDOW = 5


def run_task(*_):
    # Environment
    env = TfEnv(
        normalize(
            MultiTaskEnv(
                task_env_cls=PR2ArmClockEnv,
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
    act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
    act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
    traj_lb = np.stack([act_obs_lb] * TRAJ_ENC_WINDOW)
    traj_ub = np.stack([act_obs_ub] * TRAJ_ENC_WINDOW)
    traj_space = Box(traj_lb, traj_ub)

    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
    latent_obs_space = concat_spaces(latent_space, env.observation_space)
    env_spec_embed = EnvSpec(latent_obs_space, env.action_space)

    # Base policy
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env_spec_embed,
        hidden_sizes=(32, 32),
        adaptive_std=True,  # Must be True for embedding learning
    )

    # Embeddings
    task_embedding = GaussianMLPEmbedding(
        name="task_embedding",
        embedding_spec=task_embed_spec,
        hidden_sizes=(32, 32),
        adaptive_std=True,  # Must be True for embedding learning
    )

    traj_embedding = GaussianMLPEmbedding(
        name="traj_embedding",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(32, 32),
        adaptive_std=True,  # Must be True for embedding learning
    )

    baseline = LinearFeatureBaseline(env_spec=env_spec_embed)

    algo = TRPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        embedding=task_embedding,
        inference=traj_embedding,
        batch_size=4000,
        max_path_length=MAX_PATH_LENGTH,
        n_itr=400000000,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    exp_prefix='trpo_pr2_clock_embed',
    n_parallel=N_PARALLEL,
    plot=True,
)
