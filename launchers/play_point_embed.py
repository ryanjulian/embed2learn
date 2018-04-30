import sys
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import joblib

from sandbox.embed2learn.envs.multi_task_env import normalize, TfEnv

MAX_PATH_LENGTH = 100
SAMPLING_POSITIONS = [-0.75, -.5, 0., .5, 0.75]
COLOR_MAPS = ['autumn', 'winter']


def rollout_given_z(env,
                    agent,
                    z,
                    max_path_length=np.inf,
                    animated=False,
                    speedup=1):
    o = env.reset()
    agent.reset()

    if animated:
        env.render()

    path_length = 0
    observations = []
    while path_length < max_path_length:
        a, agent_info = agent.get_action_from_latent(o, z)
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            time.sleep(0.05 / speedup)

    return np.array(observations)


def play(pkl_filename):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(4, 4))
    with tf.Session():
        snapshot = joblib.load(pkl_filename)
        env = snapshot["env"]
        policy = snapshot["policy"]
        task_envs = env._wrapped_env._wrapped_env._task_envs
        goals = np.array([te._goal for te in task_envs])

        color_maps = [matplotlib.cm.get_cmap(cm) for cm in COLOR_MAPS]
        for task, env in enumerate(task_envs):
            plt.scatter([goals[task, 0]], [goals[task, 1]],
                        s=50, color=color_maps[task % len(color_maps)](0),
                        zorder=2, label="Task %i" % (task + 1))
            onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
            onehot[task] = 1
            z, latent_info = policy.get_latent(onehot)
            for i, x in enumerate(SAMPLING_POSITIONS):
                # systematic sampling of latent from embedding distribution
                z = latent_info["mean"] + x * np.exp(latent_info["log_std"])
                obs = rollout_given_z(TfEnv(normalize(env)), policy, z,
                                      max_path_length=MAX_PATH_LENGTH, animated=False)
                plt.plot(obs[:, 0], obs[:, 1], alpha=0.7,
                         color=color_maps[task % len(color_maps)](i * 1. / len(SAMPLING_POSITIONS)))
        plt.grid(True)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.axes().set_aspect('equal')
        plt.legend()
        plt.tight_layout()
        plt.savefig('rollout.pdf')
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: %s PKL_FILENAME' % sys.argv[0])
        sys.exit(0)

    play(sys.argv[1])
