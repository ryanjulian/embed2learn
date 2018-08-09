from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from garage.tf.algos import TRPO
from garage.envs import normalize
from garage.envs.env_spec import EnvSpec
from garage.misc.instrument import run_experiment
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from sandbox.embed2learn.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.embed2learn.envs.multiworld import FlatTorqueReacher

# SimpleReacherEnv
# GOALS = [
#   # (  ?,    x,   ?)
#     (0.3, -0.3, 0.3),
#     (0.3, 0.3, 0.3),
#     (0.3, 0.3, 0.4),
#     (0.3, -0.3, 0.3),  # confusion goal
# ]

# FlatTorqueReacherEnv
GOALS = [
    # (  ?,    ?,   ?)
    # (0.05, 0.6, 0.15),
    (0.3, 0.6, 0.15),
    (-0.3, 0.6, 0.15),
]


def run_task(v):
    v = SimpleNamespace(**v)

    # Environment
    env = FlatTorqueReacher(
        fix_goal=True,
        fixed_goal=GOALS[0],
        reward_type="hand_distance",
        hand_distance_completion_bonus=0.,
        torque_limit_pct=0.2,
        indicator_threshold=0.03,
        velocity_penalty_coeff=0.01,
        action_scale=10.0,
        hide_goal_pos=True,
    )
    env = TfEnv(normalize(env))

    # Policy
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(64, 32),
        init_std=v.policy_init_std,
    )

    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,  # 4096
        max_path_length=v.max_path_length,
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=True,
        #optimizer_args=dict(max_grad_norm=0.5)
    )
    algo.train()


config = dict(
    batch_size=4096,
    max_path_length=100,  # 50
    policy_init_std=0.1,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='sawyer_reach_trpo_torque',
    n_parallel=8,
    seed=1,
    variant=config,
    plot=True,
)
