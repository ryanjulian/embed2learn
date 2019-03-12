from garage.envs.mujoco.sawyer import SimpleReacherEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
import tensorflow as tf


def run_task(*_):

    sess = tf.Session()
    sess.__enter__()

    env = SimpleReacherEnv(
        goal_position=(0.5, 0, 0.15),
        control_method="position_control",
        completion_bonus=2.,
        # action_scale=0.04,
    )

    env = TfEnv(env)

    action_noise = OUStrategy(env, sigma=0.05)

    actor_net = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Actor",
        hidden_sizes=[200, 100],
        hidden_nonlinearity=tf.nn.relu,)

    critic_net = ContinuousMLPQFunction(
        env_spec=env.spec,
        name="Critic",
        hidden_sizes=[200, 100],
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
        n_rollout_steps=200,
        n_train_steps=50,
        discount=0.9,
        replay_buffer_size=int(1e6),
        min_buffer_size=int(1e4),
        exploration_strategy=action_noise,
        actor_optimizer=tf.train.AdamOptimizer,
        critic_optimizer=tf.train.AdamOptimizer)
    ddpg.train(sess=sess)


run_experiment(
    run_task,
    exp_prefix='ddpg_sawyer_reach',
    n_parallel=1,
    seed=1,
    plot=True,
)
