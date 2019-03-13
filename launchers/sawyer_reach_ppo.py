from types import SimpleNamespace

from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from sawyer.mujoco.reacher_env import SimpleReacherEnv


GOALS = [
    # (  ?,    ?,   ?)
    # (0.4, -0.3, 0.15),
    (0.6, 0.4, 0.2),
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
            # control_cost_coeff=1.0,
            action_scale=0.04,
            randomize_start_jpos=True,
            completion_bonus=0.1,
            # terminate_on_collision=True,
            collision_penalty=0.0,
        )
        env = TfEnv(env)

        # Policy
        policy = GaussianMLPPolicy(
            name="Policy",
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            std_share_network=True,
            init_std=v.policy_init_std,
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(64, 64)),
        )

        # baseline = CollisionAwareBaseline(
        #     env_spec=env.spec,
        #     regressor_args=dict(hidden_sizes=(64, 64)),
        # )

        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=v.max_path_length,
            n_itr=10000,
            discount=0.99,
            lr_clip_range=0.2,
            policy_ent_coeff=0.,
            optimizer_args=dict(batch_size=32, max_epochs=10),
        )

        runner.setup(algo, env)
        runner.train(n_epochs=10000, batch_size=v.batch_size, plot=False)


config = dict(
    batch_size=4096,
    max_path_length=150,  # 50
    policy_init_std=1,  # 1.0
)

run_experiment(
    run_task,
    exp_prefix='sawyer_reach_ppo_position',
    n_parallel=1,
    seed=1,
    variant=config,
    plot=True,
)
