from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.samplers import BatchSampler

from embed2learn.envs import PointEnv

def run_task(*_):
    with LocalRunner() as runner:
        env = TfEnv(normalize(PointEnv(goal=(-1, 0))))

        policy = GaussianMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            max_kl_step=0.01,
        )

        batch_size = 4000
        max_path_length = 100
        n_envs = batch_size // max_path_length

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=batch_size, plot=False)

run_experiment(
    run_task,
    snapshot_mode="last",
    seed=1,
)
