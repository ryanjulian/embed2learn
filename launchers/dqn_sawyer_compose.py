import os.path as osp
import os
from datetime import datetime

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
from baselines import logger
from garage.config import LOG_DIR
from garage.envs.mujoco.sawyer import SimpleReacherEnv
import joblib
import tensorflow as tf
import numpy as np

from embed2learn.envs.discrete_embedded_policy_env import DiscreteEmbeddedPolicyEnv


# USE_LOG = "local/sawyer-reach-embed-8goal/sawyer_reach_embed_8goal_2018_08_19_17_09_21_0001/"
USE_LOG = "local/sawyer-reach-embed-notanh/sawyer_reach_embed_notanh_2018_08_23_12_38_13_0001"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_300.pkl")


def main():
    seed = 1

    sess = tf.Session()
    sess.__enter__()

    inner_env = SimpleReacherEnv(
        goal_position=(0.5, 0, 0.15),
        control_method="position_control",
        completion_bonus=2.,
        action_scale=0.04,
    )
    latent_policy = joblib.load(latent_policy_pkl)["policy"]
    ntasks = latent_policy.task_space.shape[0]
    tasks = np.eye(ntasks)
    latents = [latent_policy.get_latent(tasks[t])[1]["mean"] for t in range(ntasks)]
    print("Latents:\n\t", "\n\t".join(map(str, latents)))

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = osp.join(LOG_DIR, "dqn-compose-reach", "dqn-%s" % timestamp)

    ckpt_dir = osp.join(log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    env = DiscreteEmbeddedPolicyEnv(inner_env, latent_policy, latents)
    logger.configure(dir=log_dir, format_strs=["stdout","log","csv","tensorboard"])
    set_global_seeds(seed)
    env = bench.Monitor(env, logger.get_dir())

    act = deepq.learn(
        env,
        "mlp",
        hiddens=[64, 64],
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        param_noise=True,
        train_freq=4,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=5,
        checkpoint_path=ckpt_dir,
        print_freq=10,
        # callback=callback
    )

    pkl_filename = osp.join(log_dir, "composer.pkl")
    print("Saving model to %s." % pkl_filename)
    act.save(pkl_filename)

    env.close()


if __name__ == '__main__':
    main()
