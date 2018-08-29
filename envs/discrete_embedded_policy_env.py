import time

import gym
import numpy as np

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from sandbox.embed2learn.policies import MultitaskPolicy


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
        return r

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def close(self):
        return self._wrapped_env.close()
