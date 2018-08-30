import os.path as osp

import time
from queue import PriorityQueue

import gym
import joblib
import numpy as np
import tensorflow as tf

from scipy.optimize import brute

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from sandbox.embed2learn.policies import MultitaskPolicy

from garage.envs.mujoco.sawyer import SimplePushEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
from garage.envs.mujoco.sawyer import PushEnv


# USE_LOG = "push_embed/sawyer_pusher_rel_obs_embed_udlr_2018_08_27_15_49_32_0001"
# # USE_LOG = "push_embed/sawyer_pusher_rel_obs_embed_udlr_2018_08_23_15_32_40_0001"
# LOG_DIR = "/home/eric/.deep-rl-docker/garage_embed/data"
# latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_600.pkl")
USE_LOG = "push_embed/sawyer_pusher_rel_obs_embed_udlr_2018_08_23_15_32_40_0001"
LOG_DIR = "/home/eric/.deep-rl-docker/garage_embed/data"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_596.pkl")

GOAL_SEQUENCE = np.array([[0.10, -0.10, 0.03],
                         [0.10, 0.10, 0.03],
                         [0., -0.10, 0.03],
                         [0., 0., 0.03]])

PATH_LENGTH = 500  # 80
SKIP_STEPS = 15  # 20

SEARCH_METHOD = "ucs"  # "greedy"  # "brute"

ITERATIONS = PATH_LENGTH // SKIP_STEPS


# XXX I'm using Hejia's garage.zip to get his SimplePushEnv
class DiscreteEmbeddedPolicyEnv(gym.Env, Parameterized):
    """Discrete action space where each action corresponds to one latent."""

    def __init__(self,
                 wrapped_env=None,
                 wrapped_policy=None,
                 latents=None,
                 skip_steps=1,
                 deterministic=True):
        assert isinstance(wrapped_policy, MultitaskPolicy)
        assert isinstance(latents, list)
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._wrapped_env = wrapped_env
        self._wrapped_policy = wrapped_policy
        self._latents = latents
        self._last_obs = None
        self._skip_steps = skip_steps
        self._deterministic = deterministic

    def reset(self, **kwargs):
        self._last_obs = self._wrapped_env.reset(**kwargs)
        self._wrapped_policy.reset()
        return self._last_obs

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._latents))

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, action, animate=False, markers=[]):
        latent = self._latents[action]
        accumulated_r = 0
        # print("action", action)
        for _ in range(self._skip_steps):
            action, agent_info = self._wrapped_policy.get_action_from_latent(
                latent, np.copy(self._last_obs))
            # a = action
            if self._deterministic:
                a = agent_info['mean']
            else:
                a = action
            if animate:
                for m in markers:
                    self._wrapped_env.env.get_viewer().add_marker(**m)
                self._wrapped_env.render()
                timestep = 0.05
                speedup = 1.
                time.sleep(timestep / speedup)
            # scale = np.random.normal()
            # a += scale * 0.
            obs, reward, done, info = self._wrapped_env.step(a)
            accumulated_r += reward
            self._last_obs = obs
        return Step(obs, reward, done, **info)

    def set_sequence(self, actions):
        """Resets environment deterministically to sequence of actions."""

        assert self._deterministic
        self.reset()
        reward = 0
        for a in actions:
            _, r, _, _ = self.step(a)
            reward += r
        return reward

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def close(self):
        return self._wrapped_env.close()


class SequencePusherEnv(PushEnv):
    def __init__(self, sequence=None, **kwargs):
        self._sequence = np.array([subgoal for subgoal in sequence])
        self._reached = 0
        self._n_goals = len(sequence)
        PushEnv.__init__(
            self,
            delta=self._sequence[self._reached],
            **kwargs
        )

    def step(self, action):
        obs, _, _, info = super(SequencePusherEnv, self).step(action)

        current_goal = self._sequence[self._reached]
        d = np.linalg.norm(self.object_position - current_goal, axis=-1)
        reward = -d

        done = self._reached == len(self._sequence) - 1
        if d < self._distance_threshold and self._reached < len(self._sequence) - 1:
            self._reached += 1
        info['n_reached_goal'] = self._reached

        reward += self._reached

        return obs, reward, done, info

    def reset(self):
        obs = super(SequencePusherEnv, self).reset()
        self._reached = 0
        return obs


class SimpleSequencePusherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(SequencePusherEnv(*args, **kwargs))


def greedy(env: DiscreteEmbeddedPolicyEnv, ntasks: int):
    sequence = []
    for l in range(ITERATIONS):
        best_r, best_a = -np.inf, 0
        for a in range(ntasks):
            env.set_sequence(sequence)
            obs, r, done, info = env.step(a)
            if r > best_r:
                best_r, best_a = r, a
        sequence.append(best_a)
    return sequence


def ucs(env: DiscreteEmbeddedPolicyEnv, ntasks: int):
    queue = PriorityQueue()
    for a in range(ntasks):
        r = env.set_sequence([a])
        queue.put((-r, [a]))
    while not queue.empty():
        curr_r, curr_s = queue.get()
        print(curr_r, curr_s)
        if len(curr_s) == ITERATIONS:
            return curr_s
        for a in range(ntasks):
            seq = curr_s + [a]
            r = env.set_sequence(seq)
            queue.put((-r, curr_s + [a]))
    return []


def main():
    sess = tf.Session()
    sess.__enter__()

    snapshot = joblib.load(latent_policy_pkl)
    latent_policy = snapshot["policy"]
    ntasks = latent_policy.task_space.shape[0]
    tasks = np.eye(ntasks)

    latents = [latent_policy.get_latent(tasks[t])[1]["mean"] for t in range(ntasks)]
    print("Latents:\n\t", "\n\t".join(map(str, latents)))

    inner_env = SimpleSequencePusherEnv(sequence=GOAL_SEQUENCE,
                                        control_method="position_control",
                                        completion_bonus=0.,
                                        randomize_start_jpos=False,
                                        action_scale=0.04)

    env = DiscreteEmbeddedPolicyEnv(inner_env,
                                    latent_policy,
                                    latents=latents,
                                    skip_steps=SKIP_STEPS,
                                    deterministic=True)

    if SEARCH_METHOD == "brute":
        def f(x):
            env.reset()
            reward = 0.
            for i in range(ITERATIONS):
                obs, r, done, info = env.step(int(x[i]))
                reward += r
            print(x, "\tr:", reward)
            return -reward  # optimizers minimize by default

        print("Brute-forcing", ntasks ** ITERATIONS, "combinations...")
        ranges = (slice(0, ntasks, 1),) * ITERATIONS
        result = brute(f, ranges, disp=True, finish=None)
    elif SEARCH_METHOD == "greedy":
        result = greedy(env, ntasks)
    elif SEARCH_METHOD == "ucs":
        result = ucs(env, ntasks)
    else:
        raise NotImplementedError
    # result = [1] * 50

    print("Result:", result)

    src_env = snapshot["env"]
    goals = np.array(
        [te.env._goal_configuration.object_pos for te in src_env.env._task_envs])

    initial_block_pos = np.array([0.64, 0.22, 0.03])
    # markers = [dict(
    #     pos=initial_block_pos + GOAL_SEQUENCE,
    #     size=0.01 * np.ones(3),
    #     label="Goal",
    #     rgba=np.array([1., 0.8, 0., 1.])
    # )]

    markers = []
    for i, g in enumerate(GOAL_SEQUENCE):
        markers.append(dict(
            pos=initial_block_pos + g,
            size=0.01 * np.ones(3),
            label="Goal {}".format(i + 1),
            rgba=np.array([1., 0.2, 0., 1.])
        ))

    for i, g in enumerate(goals):
        markers.append(dict(
            pos=g,
            size=0.01 * np.ones(3),
            label="Goal {}".format(i + 1),
            rgba=np.array([1., 0.2, 0., 1.])
        ))

    while True:
        env.reset()
        reward = 0.
        for i in range(ITERATIONS):
            obs, r, done, info = env.step(int(result[i]),
                                          animate=True,
                                          markers=markers)
            reward += r
        print(result, "\tr:", reward)


if __name__ == "__main__":
    main()
