import multiprocessing as mp
import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.env_spec import EnvSpec

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer

from sandbox.embed2learn.algos.trpo_task_embedding import TRPOTaskEmbedding
from sandbox.embed2learn.embeddings.gaussian_mlp_embedding import GaussianMLPEmbedding
from sandbox.embed2learn.embeddings.one_hot_embedding import OneHotEmbedding
from sandbox.embed2learn.embeddings.embedding_spec import EmbeddingSpec
from sandbox.embed2learn.envs.point_env import PointEnv
from sandbox.embed2learn.envs.multi_task_env import MultiTaskEnv
from sandbox.embed2learn.envs.multi_task_env import TfEnv
from sandbox.embed2learn.envs.multi_task_env import normalize
from sandbox.embed2learn.embeddings.utils import concat_spaces

TASKS = {
    '(-1, 0)': {'args': [], 'kwargs': {'goal': (-1, 0)}},
    '(1, 0)': {'args': [], 'kwargs': {'goal': (1, 0)}},
} # yapf: disable
TASK_NAMES = sorted(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]

# NOTE: trajectory encoder network size is O(n) with MAX_PATH_LENGTH
MAX_PATH_LENGTH = 20
LATENT_LENGTH = 4
N_PARALLEL = 1

#TODO: there seems to be a locking problems somewhere for n>1
# Choose number of cores for sampling
#N_PARALLEL = max(2, mp.cpu_count() - 4)


def run_task(*_):
    # Environment
    env = TfEnv(
        normalize(
            MultiTaskEnv(
                task_env_cls=PointEnv,
                task_args=TASK_ARGS,
                task_kwargs=TASK_KWARGS)))

    # Latent space and embedding specs
    # TODO: this should probably be done in Embedding or Algo
    latent_lb = np.zeros(LATENT_LENGTH, )
    latent_ub = np.ones(LATENT_LENGTH, )
    latent_space = Box(latent_lb, latent_ub)

    # trajectory space is MAX_PATH_LENGTH actions and states
    act_lb, act_ub = env.action_space.bounds
    act_lb_flat = env.action_space.flatten(act_lb)
    act_ub_flat = env.action_space.flatten(act_ub)
    obs_lb, obs_ub = env.observation_space.bounds
    obs_lb_flat = env.observation_space.flatten(obs_lb)
    obs_ub_flat = env.observation_space.flatten(obs_ub)
    act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
    act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
    traj_lb = np.tile(act_obs_lb, MAX_PATH_LENGTH)
    traj_ub = np.tile(act_obs_ub, MAX_PATH_LENGTH)
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
    # task_embedding = OneHotEmbedding(
    #     name="task_embedding",
    #     embedding_spec=task_embed_spec,
    # )

    traj_embedding = GaussianMLPEmbedding(
        name="traj_embedding",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(32, 32),
        adaptive_std=True,  # Must be True for embedding learning
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    task_opt = FirstOrderOptimizer(
        name='task_enc_opt', tf_optimizer_args={
            'learning_rate': 1e-5 # 1e-3 causes instability
    })
    traj_opt = FirstOrderOptimizer(
        name='traj_enc_opt', tf_optimizer_args={
            'learning_rate': 1e-5 # 1e-3 causes instability
    })

    algo = TRPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        task_encoder=task_embedding,
        trajectory_encoder=traj_embedding,
        batch_size=800,
        max_path_length=MAX_PATH_LENGTH,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=False,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=1e-3,
        task_encoder_ent_coeff=1e-3,
        trajectory_encoder_ent_coeff=1e-3,
        #task_encoder_optimizer=task_opt,
        #trajectory_encoder_optimizer=traj_opt,
    )
    algo.train()


run_experiment_lite(
    run_task,
    exp_prefix='trpo_point_embed',
    n_parallel=N_PARALLEL,
    plot=False,
)