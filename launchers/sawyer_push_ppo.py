from types import SimpleNamespace

from garage.tf.algos import PPO
from garage.envs.mujoco.sawyer.pusher_env import SimplePusherEnv
from garage.misc.instrument import run_experiment
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv

from embed2learn.policies.gaussian_mlp_policy import GaussianMLPPolicy


def run_task(v):
    v = SimpleNamespace(**v)

    # Environment
    env = SimplePusherEnv(action_scale=0.04, control_method="position_control", completion_bonus=0.1, collision_penalty=0.05)

    env = TfEnv(env)

    # Policy
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(256,128),
        init_std=v.policy_init_std,
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(hidden_sizes=(256,128)),
    )

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,  # 4096
        max_path_length=v.max_path_length,
        n_itr=2000,
        discount=0.99,
        step_size=0.2,
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
    exp_prefix='sawyer_pusher_ppo_done',
    n_parallel=4,
    seed=1,
    variant=config,
    plot=True,
)
