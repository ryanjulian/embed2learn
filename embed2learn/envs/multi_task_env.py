import random

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step


class MultiTaskEnv(Env, Serializable):
    def __init__(self, wrapped_env_cls, all_args, all_kwargs):
        Serializable.quick_init(self, locals())
        self._wrapped_envs = [
            wrapped_env_cls(*args, **kwargs)
            for args, kwargs in zip(all_args, all_kwargs)
        ]
        self._current_env = None

        self._choose_current_env()

    @property
    def wrapped_envs(self):
        return self._wrapped_envs

    def reset(self, **kwargs):
        self._choose_current_env()
        return self._obs_with_one_hot(self._current_env.reset(**kwargs))

    @property
    def action_space(self):
        return self._wrapped_envs[0].action_space

    @property
    def observation_space(self):
        n = len(self._wrapped_envs)
        one_hot_ub = np.ones(n)
        one_hot_lb = np.zeros(n)
        env_ub, env_lb = self._wrapped_envs[0].observation_space.bounds
        return spaces.Box(
            np.concatenate([one_hot_ub, env_ub]),
            np.concatenate([one_hot_lb, env_lb]))

    def step(self, action):
        obs, reward, done, kwargs = self._current_env.step(action)
        oh_obs = self._obs_with_one_hot(obs)
        return Step(oh_obs, reward, done, **kwargs)

    def render(self, *args, **kwargs):
        return self._current_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        return self._current_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_envs[0].horizon

    def terminate(self):
        for env in self._wrapped_envs:
            env.terminate()

    def get_param_values(self):
        return self._current_env.get_param_values()

    def set_param_values(self, params):
        self._current_env.set_param_values(params)

    def _obs_with_one_hot(self, obs):
        one_hot = np.zeros(len(self._wrapped_envs))
        one_hot[self._wrapped_envs.index(self._current_env)] = 1
        oh_obs = np.concatenate([one_hot, obs])
        return oh_obs

    def _choose_current_env(self):
        # uniform random
        #return random.choice(self._wrapped_envs)

        # round robin
        if self._current_env is None:
            self._current_env = self._wrapped_envs[0]
        else:
            i = self._wrapped_envs.index(self._current_env)
            n = len(self._wrapped_envs)
            self._current_env = self._wrapped_envs[(i + 1) % n]
