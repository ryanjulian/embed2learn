from garage.algos import TRPO
from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.dm_control import DmControlEnv
from garage.experiment import LocalRunner, run_experiment
from garage.policies import GaussianMLPPolicy


def run_task(*_):

    with LocalRunner() as runner:
        env = normalize(DmControlEnv.from_suite('cartpole', 'balance'))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=400, batch_size=4000, plot=True)


run_experiment(
    run_task,
    snapshot_mode="last",
)
