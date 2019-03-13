from types import SimpleNamespace

from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from sawyer.mujoco.reacher_env import SimpleReacherEnv


GOALS = [
    # (  ?,    ?,   ?)
    (0.6, 0.3, 0.3),
    # (0.3, 0.6, 0.15),
    # (-0.3, 0.6, 0.15),
]


def run_task(v):
    v = SimpleNamespace(**v)

    with LocalRunner() as runner:
        # Environment
        env = SimpleReacherEnv(
            goal_position=GOALS[0],
            control_method="position_control",
            completion_bonus=5
        )

        env = TfEnv(env)

        # Policy
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(64, 32),
            init_std=v.policy_init_std,
        )

        baseline = GaussianMLPBaseline(env_spec=env.spec)

        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=v.max_path_length,
            discount=0.99,
            lr_clip_range=0.2,
            optimizer_args=dict(batch_size=32, max_epochs=10),
            plot=True,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=1000, batch_size=v.batch_size, plot=False)


config = dict(
    batch_size=4096,
    max_path_length=100,  # 50
    policy_init_std=0.1,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='sawyer_reach_ppo_position',
    n_parallel=4,
    seed=1,
    variant=config,
    plot=True,
)
