import os.path as osp

import joblib
import tensorflow as tf
from tqdm import tqdm

from garage.config import LOG_DIR
from garage.envs.mujoco.sawyer import SimpleReacherEnv
from garage.misc.instrument import run_experiment
from garage.tf.algos import DDPG
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.envs import TfEnv
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction

from sandbox.embed2learn.envs.embedded_policy_env import EmbeddedPolicyEnv
from sandbox.embed2learn.envs.mujoco.sequence_reacher import SimpleReacherSequenceEnv
from sandbox.embed2learn.policies.mpc_policy import MPCPolicy 

latent_policy_pkl = "/home/zhanpenghe/Desktop/exp/rebase/garage/data/local/sawyer-reach-embed-8goal/sawyer_reach_embed_8goal_2018_08_19_17_09_21_0001/itr_180.pkl"

goal_sequence = [
    (0.5, 0, 0.3),
    (0.5, -0.3, 0.3),
    (0.7, -0.2, 0.1),
    (0.7, 0, 0.1),
    (0.7,  0.2, 0.1),
    (0.5,  0.3, 0.3),
    (0.5, 0, 0.3),
]

def rollout_mpc(env, mpc_policy, max_path_length=500):
    curr_reached = 0
    o = env.reset()
    for _ in tqdm(range(max_path_length)):
        env.render()
        a = mpc_policy.get_action(o, env.env._wrapped_env.env.get_state())
        next_o, r, done, info = env.step(a)
        reached = info['n_reached_goal']
        if reached > curr_reached:
            curr_reached = reached
            print("Reached {}".format(curr_reached))
        o = next_o
        if done:
            break

def play():
    sess = tf.Session()
    sess.__enter__()

    inner_env = SimpleReacherSequenceEnv(
        sequence=goal_sequence,
        control_method="position_control",
        subgoal_bonus=30.,
        completion_bonus=300,
        action_scale=0.04,
    )

    inner_env_2 = SimpleReacherSequenceEnv(
        sequence=goal_sequence,
        control_method="position_control",
        subgoal_bonus=30.,
        completion_bonus=300,
        action_scale=0.04,
    )
    latent_policy = joblib.load(latent_policy_pkl)["policy"]

    env = TfEnv(EmbeddedPolicyEnv(inner_env, latent_policy))

    policy = MPCPolicy(
        embedding=latent_policy._embedding, 
        n_learned_skills=8, 
        inner_policy=latent_policy, 
        inner_env=inner_env_2,
    )

    while True:
        rollout_mpc(
            env=env,
            mpc_policy=policy,
        )


if __name__ == '__main__':
    play()