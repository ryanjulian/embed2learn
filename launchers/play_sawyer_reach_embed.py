import argparse
import json
import os.path as osp
import sys
import time

import joblib
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial import ConvexHull
import tensorflow as tf

from sandbox.embed2learn.envs.util import colormap_mpl


def rollout(env,
            agent,
            z,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):

    observations = []
    tasks = []
    latents = []
    latent_infos = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    # Resets
    o = env.reset()
    agent.reset()

    # Sample embedding network
    # NOTE: it is important to do this _once per rollout_, not once per
    # timestep, since we need correlated noise.
    # t = env.active_task_one_hot
    # z, latent_info = agent.get_latent(t)

    if animated:
        env.render()

    path_length = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action_from_latent(z, o)
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return np.array(observations)


def get_z_dist(t, policy):
    """ Get the latent distribution for a task """
    onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
    onehot[t] = 1
    _, latent_info = policy.get_latent(onehot)
    return latent_info["mean"], np.exp(latent_info["log_std"])


def play(pkl_file):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        env = snapshot["env"]
        policy = snapshot["policy"]

        # Tasks and goals
        num_tasks = policy.task_space.flat_dim
        task_envs = env._wrapped_env.env._task_envs
        # goals = np.array([te._goal_position for te in task_envs])
        task_cmap = colormap_mpl(num_tasks)

        # Embedding distributions
        z_dists = [get_z_dist(t, policy) for t in range(num_tasks)]
        z_means = np.array([d[0] for d in z_dists])
        z_stds = np.array([d[1] for d in z_dists])

        while True:
            # Render individual task policies
            for t in range(num_tasks):
                z = z_means[t]

                # Run rollout
                print("Animating task {}".format(t + 1))
                rollout(
                    task_envs[t],
                    policy,
                    z,
                    max_path_length=150,
                    animated=True)

            # Render mean policy of task 1 and 2
            if num_tasks > 1:
                z = (z_means[0] + z_means[1]) / 2
                print("Animating mean of tasks {} and {}".format(1, 2))
                rollout(
                    task_envs[0],
                    policy,
                    z,
                    max_path_length=150,
                    animated=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('pkl_file', metavar='pkl_file', type=str,
                    help='.pkl file containing the policy')
    args = parser.parse_args()

    play(args.pkl_file)
