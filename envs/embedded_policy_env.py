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
        # lb = np.array([-13, -18, - 18])
        # ub = np.array([17, 18, 18])

        # Below is for point env         
        lb = np.full_like(lower, -13.0)
        ub = np.full_like(lower, 13.0)
        return gym.spaces.Box(lb, ub, dtype=np.float32)

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, latent, use_mean=True):
        for _ in range(10):
            action, agent_info = self._wrapped_policy.get_action_from_latent(
                latent, self._last_obs)
            if use_mean:
                a = agent_info['mean']
            else:
                a = action
            scale = np.random.normal()
            a += scale * 0.02
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


class AlmostContinuousEmbeddedPolicyEnv(gym.Env, Parameterized):
    # TODO: inherit EmbededPolicyEnv, RENAME THIS CLASS
    def __init__(self, wrapped_env=None, wrapped_policy=None):
        assert isinstance(wrapped_policy, MultitaskPolicy)
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._wrapped_env = wrapped_env
        self._wrapped_policy = wrapped_policy
        self._last_obs = None

        n_task = self._wrapped_policy.task_space.flat_dim
        one_hots = np.identity(n_task)
        latents, infos = self._wrapped_policy._embedding.get_latents(one_hots)
        latents_means = infos["mean"]
        self._latents_combination_hash = list()
        for i in range(n_task):
            for j in range(i+1, n_task):
                self._latents_combination_hash.append((latents_means[i, ...], latents_means[j, ...]))

        self._latents_combination_hash = tuple(self._latents_combination_hash)
        self._n_skills = n_task

    def reset(self, **kwargs):
        self._last_obs = self._wrapped_env.reset(**kwargs)
        return self._last_obs

    @property
    def action_space(self):
        # A super wierd looking action space
        lb = np.array([0., 0.])
        ub = np.array([self._n_skills * (self._n_skills-1) / 2, 1.])
        return gym.spaces.Box(lb, ub, dtype=np.float32)

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def step(self, action, use_mean=True):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        idx = int(action[0])
        if idx == self.action_space.high:
            idx -= 1
        latents = self._latents_combination_hash[idx]
        latent = action[1] * latents[0] + (1 - action[1]) * latent[1]
        # TODO: Make this step size a param..
        for _ in range(10):
            action, agent_info = self._wrapped_policy.get_action_from_latent(
                latent, self._last_obs)
            if use_mean:
                a = agent_info['mean']
            else:
                a = action
            scale = np.random.normal()
            a += scale * 0.02
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
