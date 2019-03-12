from types import SimpleNamespace

from akro.tf import Box
from garage.envs.env_spec import EnvSpec
from garage.experiment import run_experiment
import numpy as np

from embed2learn.algos import PPOTaskEmbedding
from embed2learn.baselines import MultiTaskGaussianMLPBaseline
from embed2learn.embeddings import GaussianMLPEmbedding
from embed2learn.policies import GaussianMLPMultitaskPolicy
from embed2learn.embeddings import EmbeddingSpec
from embed2learn.envs.multiworld import FlatTorqueReacher
from embed2learn.envs import MultiTaskEnv
from embed2learn.envs import TfEnv
from embed2learn.envs import normalize
from embed2learn.embeddings.utils import concat_spaces


# FlatTorqueReacherEnv
GOALS = [
  # (L/R, depth, height)
    (0.3, 0.6, 0.15),
    (-0.3, 0.6, 0.15),
]

# FlatXYZReacher/FlatTorqueReacher
TASKS = {
    str(t + 1): {
        'args': [],
        'kwargs': {
            'fix_goal': True,
            'fixed_goal': g,
            'reward_type': 'hand_distance',
            'torque_limit_pct': 0.2,
            'indicator_threshold': 0.03,
            'velocity_penalty_coeff': 0.01,
            'action_scale': 10.0,
            'hide_goal_pos': True,
        }
    }
    for t, g in enumerate(GOALS[:2])
}


def run_task(v):
    v = SimpleNamespace(**v)

    task_names = sorted(v.tasks.keys())
    task_args = [v.tasks[t]['args'] for t in task_names]
    task_kwargs = [v.tasks[t]['kwargs'] for t in task_names]

    # Environment
    env = TfEnv(
            normalize(
              MultiTaskEnv(
                task_env_cls=FlatTorqueReacher,
                task_args=task_args,
                task_kwargs=task_kwargs)))

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

    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
    task_obs_space = concat_spaces(env.task_space, env.observation_space)
    env_spec_embed = EnvSpec(task_obs_space, env.action_space)

    # TODO(): rename to inference_network
    traj_embedding = GaussianMLPEmbedding(
        name="inference",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(64, 64),
        std_share_network=True,
        init_std=1.0,
    )

    # Embeddings
    task_embedding = GaussianMLPEmbedding(
        name="embedding",
        embedding_spec=task_embed_spec,
        hidden_sizes=(64, 64),
        std_share_network=True,
        init_std=v.embedding_init_std,  # 1.0
        max_std=v.embedding_max_std,  # 2.0
        # std_parameterization="softplus",
    )

    # Multitask policy
    policy = GaussianMLPMultitaskPolicy(
        name="policy",
        env_spec=env.spec,
        task_space=env.task_space,
        embedding=task_embedding,
        hidden_sizes=(64, 32),
        std_share_network=True,
        init_std=v.policy_init_std,
        max_std=v.policy_max_std,
        # std_parameterization="softplus",
    )

    # baseline = MultiTaskLinearFeatureBaseline(env_spec=env_spec_embed)
    extra = v.latent_length + len(v.tasks)
    baseline = MultiTaskGaussianMLPBaseline(
        env_spec=env.spec, extra_dims=extra)

    algo = PPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        inference=traj_embedding,
        batch_size=v.batch_size,  # 4096
        max_path_length=v.max_path_length,
        n_itr=1000,
        discount=0.99,
        step_size=0.2,
        plot=True,
        policy_ent_coeff=v.policy_ent_coeff,
        embedding_ent_coeff=v.embedding_ent_coeff,
        inference_ce_coeff=v.inference_ce_coeff,
        #optimizer_args=dict(max_grad_norm=0.5)
    )
    algo.train()

config = dict(
    tasks=TASKS,
    latent_length=3,  # 3
    inference_window=6,  # 6
    batch_size=4096 * len(TASKS),  # 4096 * len(TASKS)
    policy_ent_coeff=1e-5,  # 1e-2 #
    embedding_ent_coeff=3e-4,  # 1e-3
    inference_ce_coeff=2e-5,  # 1e-4
    max_path_length=100,  # 100
    embedding_init_std=1.0,  # 1.0
    embedding_max_std=2.0,  # 2.0
    policy_init_std=0.1,  # 1.0
    policy_max_std=0.2,  # 2.0
)

run_experiment(
    run_task,
    exp_prefix='sawyer_reach_multiworld_torque',
    n_parallel=12,
    seed=1,
    variant=config,
    plot=True,
)
