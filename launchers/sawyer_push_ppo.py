from types import SimpleNamespace

from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from sawyer.mujoco.pusher_env import SimplePusherEnv

from embed2learn.policies import GaussianMLPPolicy


def run_task(v):
    with LocalRunner() as runner:
        v = SimpleNamespace(**v)

        # Environment
        env = SimplePusherEnv(
            action_scale=0.04,
            control_method="position_control",
            completion_bonus=0.1,
            collision_penalty=0.05
        )

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
            discount=0.99,
            lr_clip_range=0.2,
            optimizer_args=dict(batch_size=32, max_epochs=10),
        )

        runner.setup(algo, env)
        runner.train(n_epochs=2000, batch_size=v.batch_size)


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
