from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.envs import TfEnv

from embed2learn.envs.mujoco import PR2ArmClockEnv
from embed2learn.envs import OneHotMultiTaskEnv

TASKS = {
    'center': {'args': [], 'kwargs': {'target': 'center'}},
    'hour_12': {'args': [], 'kwargs': {'target': 'hour_12'}},
    'hour_1': {'args': [], 'kwargs': {'target': 'hour_1'}},
    'hour_2': {'args': [], 'kwargs': {'target': 'hour_2'}},
    'hour_3': {'args': [], 'kwargs': {'target': 'hour_3'}},
    'hour_4': {'args': [], 'kwargs': {'target': 'hour_4'}},
    'hour_5': {'args': [], 'kwargs': {'target': 'hour_5'}},
    'hour_6': {'args': [], 'kwargs': {'target': 'hour_6'}},
    'hour_7': {'args': [], 'kwargs': {'target': 'hour_7'}},
    'hour_8': {'args': [], 'kwargs': {'target': 'hour_8'}},
    'hour_9': {'args': [], 'kwargs': {'target': 'hour_9'}},
    'hour_10': {'args': [], 'kwargs': {'target': 'hour_10'}},
    'hour_11': {'args': [], 'kwargs': {'target': 'hour_11'}},
} # yapf: disable
TASK_NAMES = sorted(TASKS.keys())
TASK_ARGS = [TASKS[t]['args'] for t in TASK_NAMES]
TASK_KWARGS = [TASKS[t]['kwargs'] for t in TASK_NAMES]


def run_task(*_):
    env = TfEnv(
        normalize(
            OneHotMultiTaskEnv(
                task_env_cls=PR2ArmClockEnv,
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
        n_itr=400000000,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
