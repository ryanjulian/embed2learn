import argparse
import time

import joblib
import numpy as np
import tensorflow as tf


def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False,
            goal_markers=None):

    observations = []
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
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(agent_info['mean'])
        print(a[3])
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

def play(pkl_file):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        env = snapshot["env"]
        policy = snapshot["policy"]

        # Render individual task policies
        while True:

            # Run rollout
            print("Animating...")
            rollout(
                env,
                policy,
                max_path_length=300,
                animated=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('pkl_file', metavar='pkl_file', type=str,
                    help='.pkl file containing the policy')
    args = parser.parse_args()

    play(args.pkl_file)
