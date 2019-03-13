from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.tf.algos import TRPO
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.envs import TfEnv
from sawyer.mujoco.reacher_env import SimpleReacherEnv


with LocalRunner() as runner:
    env = TfEnv(normalize(SimpleReacherEnv(goal_position=[0.3, 0.3, 0.3])))

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

    runner.setup(algo, env)
    runner.train(n_epochs=100, batch_size=4000, plot=True)
