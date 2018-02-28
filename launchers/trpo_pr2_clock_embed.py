import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.spaces.box import Box

from embed2learn.algos.trpo_with_task_embedding import TRPOWithTaskEmbedding
from embed2learn.embeddings.gaussian_mlp_embedding import GaussianMLPEmbedding
from embed2learn.embeddings.embedding_spec import EmbeddingSpec
from embed2learn.envs.mujoco.pr2_arm_clock_env import PR2ArmClockEnv
from embed2learn.envs.multi_task_env import MultiTaskEnv
from embed2learn.envs.multi_task_env import TfEnv
from embed2learn.envs.multi_task_env import normalize
from embed2learn.policies.embedded_multi_task_policy import EmbeddedMultiTaskPolicy


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

# NOTE: trajectory encoder network size is O(n) with MAX_PATH_LENGTH
MAX_PATH_LENGTH = 100


def run_task(*_):
    # Environment
    env = TfEnv(
        normalize(
            MultiTaskEnv(
                task_env_cls=PR2ArmClockEnv,
                task_args=TASK_ARGS,
                task_kwargs=TASK_KWARGS)))

    # Latent space and embedding specs
    # TODO: this should probably be done in Embedding
    latent_lb = np.zeros(13, )
    latent_ub = np.ones(13, )
    latent_space = Box(latent_lb, latent_ub)

    # trajectory space a single flattened action followed by a MAX_PATH_LENGTH
    # flattened state trajectories
    act_lb, act_ub = env.action_space.bounds
    act_lb_flat = env.action_space.flatten(act_lb)
    act_ub_flat = env.action_space.flatten(act_ub)
    obs_lb, obs_ub = env.observation_space.bounds
    obs_lb_flat = env.observation_space.flatten(obs_lb)
    obs_ub_flat = env.observation_space.flatten(obs_ub)
    traj_lb = np.tile(obs_lb_flat, MAX_PATH_LENGTH)
    traj_ub = np.tile(obs_ub_flat, MAX_PATH_LENGTH)
    traj_space = Box(
        np.concatenate([act_lb_flat, traj_lb]),
        np.concatenate([act_ub_flat, traj_ub]))
    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    traj_embed_spec = EmbeddingSpec(traj_space, latent_space)

    # Base policy
    base_policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    # Embeddings
    task_embedding = GaussianMLPEmbedding(
        name="task_embedding",
        embedding_spec=task_embed_spec,
        hidden_sizes=(32, 32),
    )

    traj_embedding = GaussianMLPEmbedding(
        name="traj_embedding",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(32, 32),
    )

    # Policy with embeddings
    policy = EmbeddedMultiTaskPolicy(
        policy=base_policy,
        task_encoder=task_embedding,
        trajectory_encoder=traj_embedding,
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPOWithTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        task_encoder=task_embedding,
        trajectory_encoder=traj_embedding,
        batch_size=4000,
        max_path_length=MAX_PATH_LENGTH,
        n_itr=400000000,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
