from rllab.baselines import LinearFeatureBaseline
from rllab.envs import normalize
from rllab.misc.instrument import stub
from rllab.misc.instrument import run_experiment_lite

from sandbox.rocky.tf.algos import TRPO
from sandbox.rocky.tf.policies import GaussianMLPPolicy
from sandbox.rocky.tf.envs import TfEnv

from sandbox.embed2learn.envs import PointEnv


def run_task(*_):
    env = TfEnv(normalize(PointEnv(goal=(-1, 0))))

    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        plot=False,
        force_batch_sampler=True,
    )
    algo.train()


run_task()
# run_experiment_lite(
#     run_task,
#     n_parallel=20,
#     plot=False,
# )
