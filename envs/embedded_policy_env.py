import gym
import numpy as np

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from sandbox.embed2learn.policies import MultitaskPolicy


class EmbeddedPolicyEnv(gym.Env, Parameterized):
    def __init__(self, wrapped_env=None, wrapped_policy=None):
        assert isinstance(wrapped_policy, MultitaskPolicy)
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._wrapped_env = wrapped_env
        self._wrapped_policy = wrapped_policy
        self._last_obs = None

    def reset(self, **kwargs):
        self._last_obs = self._wrapped_env.reset(**kwargs)
        return self._last_obs

    @property
    def action_space(self):
        # TODO: fix gym.spaces conversion
        lower, upper = self._wrapped_policy.latent_space.bounds
        lb = np.full_like(lower, -10.0)
        ub = np.full_like(lower, 10.0)
        return gym.spaces.Box(lb, ub, dtype=np.float32)

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, latent):
        a, _ = self._wrapped_policy.get_action_from_latent(
            self._last_obs, latent)
        obs, reward, done, info = self._wrapped_env.step(a)
        self._last_obs = obs
        return Step(obs, reward, done, **info)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def close(self):
        return self._wrapped_env.close()
