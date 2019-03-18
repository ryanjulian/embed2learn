import argparse
import os.path as osp

from garage.config import LOG_DIR
from garage.experiment import LocalRunner, run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.exploration_strategies import OUStrategy
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
import joblib
from sawyer.mujoco.reacher_env import SimpleReacherEnv
import tensorflow as tf

from embed2learn.envs import EmbeddedPolicyEnv


def main(latent_policy_pkl):

    def run_task(*_):

        sess = tf.Session()
        sess.__enter__()
        with LocalRunner(sess=sess) as runner:
            inner_env = SimpleReacherEnv(
                goal_position=(0.5, 0, 0.15),
                control_method="position_control",
                completion_bonus=2.,
                action_scale=0.04,
            )
            latent_policy = joblib.load(latent_policy_pkl)["policy"]

            env = TfEnv(EmbeddedPolicyEnv(inner_env, latent_policy))

            action_noise = OUStrategy(env, sigma=0.2)

            policy = ContinuousMLPPolicy(
                env_spec=env.spec,
                name="Actor",
                hidden_sizes=[64, 32],
                hidden_nonlinearity=tf.nn.relu,)

            qf = ContinuousMLPQFunction(
                env_spec=env,
                name="Critic",
                hidden_sizes=[64, 32],
                hidden_nonlinearity=tf.nn.relu)

            replay_buffer = SimpleReplayBuffer(
                env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

            algo = DDPG(
                env,
                policy=policy,
                policy_lr=1e-4,
                qf_lr=1e-3,
                qf=qf,
                plot=True,
                target_update_tau=1e-2,
                n_epochs=500,
                n_train_steps=50,
                discount=0.9,
                replay_buffer=replay_buffer,
                min_buffer_size=int(1e3),
                exploration_strategy=action_noise,
                policy_optimizer=tf.train.AdamOptimizer,
                qf_optimizer=tf.train.AdamOptimizer)
            runner.setup(algo, env)
            runner.train(n_epochs=500, plot=False, n_epoch_cycles=10)

    run_experiment(
        run_task,
        exp_prefix='ddpg_sawyer_compose',
        n_parallel=1,
        seed=1,
        plot=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--log_directory',
        required=True,
        help='The directory of a logged training.',
        type=str,)

    parser.add_argument(
        '-i',
        '--iteration',
        help='The iteration # to use for composing',
        type=int,
        default=-1,  # use params.pkl if this is -1 otherwise iter_%.pkl
    )

    args = parser.parse_args()
    log_dir = args.log_directory

    pickle_filename = 'itr_{}.pkl'.format(args.iteration) if args.iteration >= 0 else 'params.pkl'
    latent_policy_pkl = osp.join(LOG_DIR, log_dir, pickle_filename)

    main(latent_policy_pkl)
