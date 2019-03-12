import os.path as osp

from garage.config import LOG_DIR
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
import joblib
import tensorflow as tf

from embed2learn.envs.embedded_policy_env import EmbeddedPolicyEnv
from embed2learn.envs import PointEnv

USE_LOG = "local/ppo-point-embed/ppo_point_embed_2018_08_12_16_26_20_0001"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_100.pkl")


def run_task(*_):
    sess = tf.Session()
    sess.__enter__()
    latent_policy = joblib.load(latent_policy_pkl)["policy"]

    inner_env = PointEnv(goal=(1.4, 1.4),completion_bonus=100)
    env = TfEnv(EmbeddedPolicyEnv(inner_env, latent_policy))

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
