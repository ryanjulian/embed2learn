from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


def run_task(*_):
    env = PointEnv(goal=(3, 3), random_start=True)
    env = TfEnv(env)

    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        init_std=20,
        std_share_network=False,
        adaptive_std=True
    )
    baseline = GaussianMLPBaseline(env_spec=env, include_action_to_input=False)
    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1024,  # 4096
        max_path_length=50,
        n_itr=1500,
        discount=0.99,
        step_size=0.2,
        policy_ent_coeff=1e-6,
        plot=True,
        use_mpc_es=True,
    )
    algo.train(sess=sess)


run_experiment(
    run_task,
    n_parallel=4,
    exp_prefix="ppo_point_compose_test_mpc",
    seed=2,
    plot=True,
)
