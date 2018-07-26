import sys

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.misc.instrument import run_experiment

from multiworld.envs.mujoco.sawyer_xyz.pick.sawyer_pick import SawyerPickEnv
from multiworld.core.flat_goal_env import FlatGoalEnv


def run_task(*_):
    env = FlatGoalEnv(SawyerPickEnv(), obs_keys=["state_observation"])
    env = TfEnv(normalize(env))

    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=500,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
        plot=True)
    algo.train()


run_experiment(
    run_task,
    n_parallel=16,
    exp_prefix="trpo_sawyer_multiworld",
    seed=1,
    plot=True,
)
