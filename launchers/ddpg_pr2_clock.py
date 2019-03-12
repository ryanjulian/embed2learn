from garage.envs import normalize
from garage.exploration_strategies import OUStrategy
from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.policies import DeterministicMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

from embed2learn.envs.mujoco import PR2ArmClockEnv
from embed2learn.envs import OneHotMultiTaskEnv

# NOTE: NOT WORKING RIGHT NOW

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
    env = normalize(
        OneHotMultiTaskEnv(
            task_env_cls=PR2ArmClockEnv,
            task_args=TASK_ARGS,
            task_kwargs=TASK_KWARGS))

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32))

    es = OUStrategy(env_spec=env.spec)

    qf = ContinuousMLPQFunction(env_spec=env.spec)

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=32,
        max_path_length=100,
        epoch_length=4000,
        min_pool_size=10000,
        n_epochs=1000000000,
        discount=0.99,
        scale_reward=0.01,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
    )
    algo.train()


run_experiment(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    plot=True,
)
