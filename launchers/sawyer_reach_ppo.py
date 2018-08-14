from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from garage.envs import normalize
from garage.envs.mujoco.sawyer import SimpleReacherEnv
from garage.envs.env_spec import EnvSpec
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
# from garage.tf.policies import GaussianMLPPolicy
from sandbox.embed2learn.policies import  GaussianMLPPolicy


GOALS = [
    # (  ?,    ?,   ?)
    (0.4, -0.3, 0.15),
    # (0.3, 0.6, 0.15),
    # (-0.3, 0.6, 0.15),
]


def run_task(v):
    v = SimpleNamespace(**v)

    # Environment
    env = SimpleReacherEnv(
        goal_position=GOALS[0],
        control_method="position_control",
        # control_cost_coeff=1.0,
        action_scale=0.04,
        # randomize_start_jpos=True,
        completion_bonus=0.0,
        # terminate_on_collision=True,
        # collision_penalty=1.,
    )
    env = TfEnv(env)

    # Policy
    policy = GaussianMLPPolicy(
        name="Policy",
        env_spec=env.spec,
        hidden_sizes=(64, 32),
        std_share_network=True,
        init_std=v.policy_init_std,
    )

    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,  # 4096
        max_path_length=v.max_path_length,
        n_itr=10000,
        discount=0.99,
        step_size=0.2,
        policy_ent_coeff=-1.,
        optimizer_args=dict(batch_size=32, max_epochs=10),
        plot=True,
    )
    algo.train()


config = dict(
    batch_size=4096,
    max_path_length=500,  # 50
    policy_init_std=1.0,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='sawyer_reach_ppo_position_collision_det',
    n_parallel=12,
    seed=1,
    variant=config,
    plot=True,
)
