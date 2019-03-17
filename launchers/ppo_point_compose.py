import argparse
import os.path as osp

import joblib
import tensorflow as tf

from garage.config import LOG_DIR
from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

from embed2learn.envs import EmbeddedPolicyEnv
from embed2learn.envs import PointEnv



def main(latent_policy_pkl):

    def run_task(*_):
        sess = tf.Session()
        sess.__enter__()
        latent_policy = joblib.load(latent_policy_pkl)["policy"]
        with LocalRunner(sess=sess) as runner:
            inner_env = PointEnv(goal=(1.4, 1.4), completion_bonus=100)
            env = TfEnv(EmbeddedPolicyEnv(inner_env, latent_policy))

            policy = GaussianMLPPolicy(
                name="composer",
                env_spec=env.spec,
                hidden_sizes=(64, 64),
                init_std=20,
                std_share_network=False,
                adaptive_std=True
            )

            baseline = GaussianMLPBaseline(env_spec=env)

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
            runner.setup(algo, env)
            runner.train(n_epochs=600, plot=False, batch_size=1024)

    run_experiment(
        run_task,
        n_parallel=1,
        exp_prefix="ppo_point_compose",
        seed=2,
        plot=False,
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
