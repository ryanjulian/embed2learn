from types import SimpleNamespace

import tensorflow as tf

from garage.envs import normalize
from garage.envs.box2d import CartpoleEnv
from garage.envs.box2d import CartpoleSwingupEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from sandbox.embed2learn.policies import  GaussianMLPPolicy


def run_task(v):
    v = SimpleNamespace(**v)

    # Environment
    env = TfEnv(normalize(CartpoleEnv()))

    # Policy
    policy = GaussianMLPPolicy(
        name="Policy",
        env_spec=env.spec,
        hidden_sizes=(64, 32),
        std_share_network=True,
        init_std=v.policy_init_std,
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(64, 32)),
    )

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,  # 4096
        max_path_length=v.max_path_length,
        n_itr=10000,
        discount=0.99,
        step_size=0.1,
        policy_ent_coeff=0.,
        optimizer_args=dict(batch_size=32, max_epochs=10),
        plot=True,
    )
    algo.train()


config = dict(
    batch_size=3000,
    max_path_length=100,  # 50
    policy_init_std=0.1,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='cartpole_swingup_ppo',
    n_parallel=12,
    seed=1,
    variant=config,
    plot=True,
)
