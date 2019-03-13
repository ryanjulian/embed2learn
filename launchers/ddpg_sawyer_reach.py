from garage.experiment import LocalRunner, run_experiment
from garage.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from sawyer.mujoco.reacher_env import SimpleReacherEnv
import tensorflow as tf


def run_task(*_):
    with LocalRunner() as runner:
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

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

        ddpg = DDPG(
            env,
            policy=actor_net,
            policy_lr=1e-4,
            qf=critic_net,
            qf_lr=1e-3,
            replay_buffer=replay_buffer,
            target_update_tau=1e-2,
            max_path_length=200,
            n_train_steps=50,
            discount=0.9,
            min_buffer_size=int(1e4),
            exploration_strategy=action_noise,
            policy_optimizer=tf.train.AdamOptimizer,
            qf_optimizer=tf.train.AdamOptimizer)

        runner.setup(ddpg, env)
        runner.train(n_epochs=500, n_epoch_cycles=10, plot=False)


run_experiment(
    run_task,
    exp_prefix='ddpg_sawyer_reach',
    n_parallel=1,
    seed=1,
    plot=False,
)
