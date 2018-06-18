from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.misc.instrument import stub
from garage.misc.instrument import run_experiment
from garage.policies import GaussianMLPPolicy

from sandbox.embed2learn.envs.mujoco import PR2ArmEnv


def run_task(*_):
    env = normalize(PR2ArmEnv())

    policy = GaussianMLPPolicy(
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
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        plot=True,
        # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
    )
    algo.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)
