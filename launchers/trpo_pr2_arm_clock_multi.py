import numpy as np

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

from embed2learn.envs.mujoco.pr2_arm_clock_env import PR2ArmClockEnv
from embed2learn.envs.multi_task_env import MultiTaskEnv

TASKS = {
    'center': {
        'args': [],
        'kwargs': {
            'target': 'center'
        }
    },
    'hour_12': {
        'args': [],
        'kwargs': {
            'target': 'hour_12'
        }
    },
    'hour_3': {
        'args': [],
        'kwargs': {
            'target': 'hour_3'
        }
    },
    'hour_6': {
        'args': [],
        'kwargs': {
            'target': 'hour_6'
        }
    },
    'hour_9': {
        'args': [],
        'kwargs': {
            'target': 'hour_9'
        }
    },
}
TASK_NAMES = list(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]

env = TfEnv(normalize(MultiTaskEnv(PR2ArmClockEnv, TASK_ARGS, TASK_KWARGS)))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=400000000,
    discount=0.99,
    step_size=0.01,
    plot=True,
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)
algo.train()
