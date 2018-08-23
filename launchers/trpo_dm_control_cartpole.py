from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.misc.instrument import run_experiment
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from sandbox.embed2learn.policies import GaussianMLPPolicy

from garage.envs.dm_control import DmControlEnv


def run_task(*_):
    env = TfEnv(DmControlEnv(domain_name='cartpole', task_name='balance'))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=400,
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
