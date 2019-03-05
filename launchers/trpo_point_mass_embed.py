import numpy as np

from garage.baselines import LinearFeatureBaseline
from garage.misc.instrument import run_experiment
from garage.misc.ext import set_seed
from garage.envs import EnvSpec

from garage.tf.policies import GaussianMLPPolicy
from garage.tf.envs import TfEnv
from akro.tf import Box

from sandbox.embed2learn.algos import TRPOTaskEmbedding
from sandbox.embed2learn.embeddings import GaussianMLPEmbedding
from sandbox.embed2learn.embeddings import EmbeddingSpec
from sandbox.embed2learn.policies import GaussianMLPMultitaskPolicy
from sandbox.embed2learn.envs import DmControlEnv
from sandbox.embed2learn.envs import MultiTaskEnv
from sandbox.embed2learn.envs import TfEnv
from sandbox.embed2learn.envs import normalize
from sandbox.embed2learn.embeddings.utils import concat_spaces


TASKS = {
    '(-0.25, 0)': {'args': [], 'kwargs': {'goal': (-0.25, 0)}},
    '(0.25, 0)': {'args': [], 'kwargs': {'goal': (0.25, 0)}},
} # yapf: disable
TASK_NAMES = sorted(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]

# Embedding params
LATENT_LENGTH = 2
TRAJ_ENC_WINDOW = 16


def run_task(plot=True, *_):
    set_seed(0)

    # Environment (train on light point mass)
    from sandbox.embed2learn.envs import point_mass_env
    from dm_control import suite
    suite._DOMAINS["embed_point_mass"] = point_mass_env
    env = TfEnv(
        normalize(
            MultiTaskEnv(
                task_env_cls=DmControlEnv,
                task_args=[["embed_point_mass", "light"]] * len(TASK_NAMES),
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
    #act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
    #act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
    act_obs_lb = obs_lb_flat
    act_obs_ub = obs_ub_flat
    traj_lb = np.stack([act_obs_lb] * TRAJ_ENC_WINDOW)
    traj_ub = np.stack([act_obs_ub] * TRAJ_ENC_WINDOW)
    traj_space = Box(traj_lb, traj_ub)

    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
    task_obs_space = concat_spaces(env.task_space, env.observation_space)
    env_spec_embed = EnvSpec(task_obs_space, env.action_space)

    # Embeddings
    task_embedding = GaussianMLPEmbedding(
        name="task_embedding",
        embedding_spec=task_embed_spec,
        hidden_sizes=(20, 20),
        adaptive_std=True,
        init_std=0.5,  # TODO was 100
        max_std=0.6,  # TODO find appropriate value
    )

    # TODO(): rename to inference_network
    traj_embedding = GaussianMLPEmbedding(
        name="traj_embedding",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(20, 10),  # was the same size as policy in Karol's paper
        # adaptive_std=True,  # Must be True for embedding learning
        std_share_network=True,
        init_std=0.001,
    )

    # Multitask policy
    policy = GaussianMLPMultitaskPolicy(
        name="policy",
        env_spec=env.spec,
        task_space=env.task_space,
        embedding=task_embedding,
        hidden_sizes=(20, 10),
        adaptive_std=True,  # Must be True for embedding learning
        init_std=0.5,  # TODO was 100
    )

    baseline = LinearFeatureBaseline(env_spec=env_spec_embed)

    algo = TRPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        inference=traj_embedding,
        batch_size=4000,
        max_path_length=100,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
        plot=plot,
        plot_warmup_itrs=20,
        policy_ent_coeff=0.0,  # 0.001,  #0.1,
        embedding_ent_coeff=0.0,  #0.1,
        inference_ce_ent_coeff=0.,  # 0.03,  #0.1,  # 0.1,
    )
    algo.train()


run_experiment(
    run_task,
    exp_prefix='trpo_point_embed',
    n_parallel=16,
    plot=True,
)
