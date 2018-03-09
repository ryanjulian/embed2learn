from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.embed2learn.envs.dm_control_env import DmControlEnv


def run_task(*_):
    env = normalize(
            DmControlEnv(
                domain_name='cartpole', task_name='balance',
                visualize_reward=True))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=400,
        discount=0.99,
        step_size=0.01,
        plot=True,
    )
    algo.train()

run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    plot=True,
)