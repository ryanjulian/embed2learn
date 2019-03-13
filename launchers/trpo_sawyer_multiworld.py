from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import run_experiment, LocalRunner
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv
from multiworld.core.flat_goal_env import FlatGoalEnv


def run_task(*_):
    with LocalRunner() as runner:
        env = FlatGoalEnv(SawyerReachXYZEnv(), obs_keys=["state_observation"])
        env = TfEnv(normalize(env))

        policy = GaussianMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=500,
            discount=0.99,
            max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=500, batch_size=4000, plot=True)


run_experiment(
    run_task,
    n_parallel=16,
    exp_prefix="trpo_sawyer_multiworld_sawyer_reach",
    seed=1,
    plot=True,
)
