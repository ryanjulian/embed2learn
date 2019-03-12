import os.path as osp

from garage.config import LOG_DIR
from garage.envs.mujoco.sawyer.reacher_env import SimpleReacherEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
import joblib
import tensorflow as tf

from embed2learn.envs.embedded_policy_env import EmbeddedPolicyEnv


USE_LOG = "local/sawyer_reach_embed_2goal_2018_08_12_14_57_14_0001"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_382.pkl")


def run_task(*_):
    sess = tf.Session()
    sess.__enter__()
    snapshot = joblib.load(latent_policy_pkl)
    latent_policy = snapshot["policy"]
    inner_env = SimpleReacherEnv(goal_position=(0.65, 0.3, 0.3), control_method="position_control", completion_bonus=30)

    env = TfEnv(EmbeddedPolicyEnv(inner_env, latent_policy))
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env,
        hidden_sizes=(64, 64),
        init_std=20,
        # std_share_network=False,
        # adaptive_std=True
    )
    baseline = GaussianMLPBaseline(env_spec=env, include_action_to_input=False)

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1024,  # 4096
        max_path_length=100,
        n_itr=1500,
        discount=0.99,
        step_size=0.2,
        policy_ent_coeff=1e-6,
        plot=True,
    )
    algo.train(sess=sess)



run_experiment(
    run_task,
    n_parallel=4,
    exp_prefix="ppo_sawyer_compose",
    seed=2,
    plot=True,
)
