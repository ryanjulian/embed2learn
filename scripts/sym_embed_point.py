import argparse

import joblib
import tensorflow as tf
import numpy as np
import time

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
from sandbox.embed2learn.embeddings.utils import concat_spaces
from sandbox.embed2learn.envs.multi_task_env import TfEnv, NormalizedMultiTaskEnv, MultiTaskEnv
from sandbox.embed2learn.envs.point_env import PointEnv

import matplotlib.pyplot as plt

GOALS = [np.array((-1, 0)), np.array((1, 0))]


def point_rollout(env: TfEnv,
                  agent,
                  task_encoder,
                  max_path_length=np.inf,
                  animated=False,
                  speedup=1,
                  always_return_paths=False,
                  modulations=10):

    multi_task_env = env.wrapped_env.wrapped_env
    assert(isinstance(multi_task_env, MultiTaskEnv))

    env_observations = []
    observations = []
    latents = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    # env = PointEnv()
    # state = env.__getstate__()

    latent_obs_space = concat_spaces(task_encoder.latent_space,
                                     env.observation_space)

    # Resets
    o = env.reset()
    agent.reset()

    plt.interactive(True)
    fig = plt.figure()
    plt.axis('equal')

    # Sample embedding network
    # NOTE: it is important to do this _once per rollout_, not once per
    # timestep, since we need correlated noise.

    task_embeddings = []
    for task_id in range(multi_task_env.num_tasks):
        task_encoder.reset()
        # TODO implement public set_active_task(task_id) method in MultiTaskEnv
        multi_task_env._active_env = multi_task_env._task_envs[task_id]
        task_onehot = multi_task_env.active_task_one_hot
        z, latent_info = task_encoder.get_latent(task_onehot)
        task_embeddings.append(z)

    for m in range(modulations):
        # linear interpolation between two task embeddings
        alpha = round(float(m) / (modulations - 1.))
        goal = GOALS[1] * alpha + GOALS[0] * (1. - alpha)
        z = task_embeddings[1] * alpha + task_embeddings[0] * (1. - alpha)
        print(z)

        plt.scatter([goal[0]], [goal[1]], c='r', s=20, zorder=100)

        env.reset()
        multi_task_env._active_env._goal = goal
        # multi_task_env._active_env._point = (0, -1)

        state = env.__getstate__()
        point_env = multi_task_env._active_env

        path_length = 0
        path = []
        while path_length < max_path_length:
            z_o = np.concatenate([z, o])
            a, agent_info = agent.get_action(z_o)
            path.append(point_env._point)
            next_o, r, done, env_info = env.step(a)
            env_observations.append(env.observation_space.flatten(o))
            observations.append(latent_obs_space.flatten(z_o))
            latents.append(task_encoder.latent_space.flatten(z))
            rewards.append(r)
            actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if done:
                break
            o = next_o
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], alpha=.7, zorder=1, linewidth=0.5, label="%.2f" % alpha)

    sess.run(data["policy"]._l_mean.W)

    plt.legend()
    plt.savefig("embed_playback.png")
    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=20,
                        help='Max length of rollout')
    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        task_encoder = data['task_encoder']

        assert isinstance(env, TfEnv)
        # env = env.wrapped_env
        # assert isinstance(env, NormalizedMultiTaskEnv)
        # env = env.wrapped_env
        # assert isinstance(env, MultiTaskEnv)
        # envs = env._task_envs
        # assert all(isinstance(env, PointEnv) for env in envs), "This script only works with PointEnv."

        point_rollout(env, policy, task_encoder,
                      max_path_length=args.max_path_length,
                      animated=True)
