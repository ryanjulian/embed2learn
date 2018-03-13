import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from sandbox.embed2learn.envs.point_env import PointEnv
from sandbox.embed2learn.envs.one_hot_multi_task_env import OneHotMultiTaskEnv

TASKS = {
    '(-1, 0)': {'args': [], 'kwargs': {'goal': (-1, 0)}},
    '(1, 0)': {'args': [], 'kwargs': {'goal': (1, 0)}},
} # yapf: disable
TASK_NAMES = sorted(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]


def run_task(*_):
    env = TfEnv(
        normalize(
            OneHotMultiTaskEnv(
                task_env_cls=PointEnv,
                task_args=TASK_ARGS,
                task_kwargs=TASK_KWARGS)))

    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        plot=False,
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=False,
)
