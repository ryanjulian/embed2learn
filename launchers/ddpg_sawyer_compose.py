import os.path as osp

from garage.config import LOG_DIR
from garage.experiment import run_experiment
from garage.exploration_strategies import OUStrategy
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
import joblib
from sawyer.mujoco import SimpleReacherEnv
import tensorflow as tf

from embed2learn.envs import EmbeddedPolicyEnv


USE_LOG = "local/sawyer-reach-embed-8goal/sawyer_reach_embed_8goal_2018_08_19_17_09_21_0001/"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_130.pkl")


def run_task(*_):

    sess = tf.Session()
    sess.__enter__()

    inner_env = SimpleReacherEnv(
        goal_position=(0.5, 0, 0.15),
        control_method="position_control",
        completion_bonus=2.,
        action_scale=0.04,
    )
    latent_policy = joblib.load(latent_policy_pkl)["policy"]

    env = TfEnv(EmbeddedPolicyEnv(inner_env, latent_policy))

    action_noise = OUStrategy(env, sigma=0.2)

    actor_net = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Actor",
        hidden_sizes=[64, 32],
        hidden_nonlinearity=tf.nn.relu,)

    critic_net = ContinuousMLPQFunction(
        env_spec=env,
        name="Critic",
        hidden_sizes=[64, 32],
        hidden_nonlinearity=tf.nn.relu)

    ddpg = DDPG(
        env,
        actor=actor_net,
        actor_lr=1e-4,
        critic_lr=1e-3,
        critic=critic_net,
        plot=True,
        target_update_tau=1e-2,
        n_epochs=500,
        n_epoch_cycles=10,
        n_rollout_steps=100,
        n_train_steps=50,
        discount=0.9,
        replay_buffer_size=int(1e6),
        min_buffer_size=int(1e3),
        exploration_strategy=action_noise,
        actor_optimizer=tf.train.AdamOptimizer,
        critic_optimizer=tf.train.AdamOptimizer)
    ddpg.train(sess=sess)

# run_task()

run_experiment(
    run_task,
    exp_prefix='ddpg_sawyer_compose',
    n_parallel=1,
    seed=1,
    plot=True,
)
