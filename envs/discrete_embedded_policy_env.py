import gym
import numpy as np

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from sandbox.embed2learn.policies import MultitaskPolicy


class DiscreteEmbeddedPolicyEnv(gym.Env, Parameterized):
    """Discrete action space where each action corresponds to the latent mean
    of one task."""

    def __init__(self,
                 wrapped_env=None,
                 wrapped_policy=None,
                 task_latent_means=None):
        assert isinstance(wrapped_policy, MultitaskPolicy)
        assert isinstance(task_latent_means, list)
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._wrapped_env = wrapped_env
        self._wrapped_policy = wrapped_policy
        self._task_latents = task_latent_means
        self._last_obs = None

    def reset(self, **kwargs):
        self._last_obs = self._wrapped_env.reset(**kwargs)
        return self._last_obs

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._task_latents))

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, action, use_mean=True):
        latent = self._task_latents[action]
        accumulated_r = 0
        for _ in range(1):
            action, agent_info = self._wrapped_policy.get_action_from_latent(
                latent, self._last_obs)
            if use_mean:
                a = agent_info['mean']
            else:
                a = action
            # scale = np.random.normal()
            # a += scale * 0.
            obs, reward, done, info = self._wrapped_env.step(a)
            accumulated_r += reward
            self._last_obs = obs
        return Step(obs, accumulated_r, done, **info)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def close(self):
        return self._wrapped_env.close()
