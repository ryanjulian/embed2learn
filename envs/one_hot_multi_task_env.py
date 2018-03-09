import random

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides

from sandbox.embed2learn.envs.multi_task_env import MultiTaskEnv


class OneHotMultiTaskEnv(MultiTaskEnv, Serializable):
    def __init__(self, **kwargs):
        super(OneHotMultiTaskEnv, self).__init__(**kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def reset(self, **kwargs):
        obs = super(OneHotMultiTaskEnv, self).reset(**kwargs)
        return self._obs_with_one_hot(obs)

    @property
    @overrides
    def observation_space(self):
        task_ub, task_lb = self.task_space.bounds
        env_ub, env_lb = super(OneHotMultiTaskEnv,
                               self).observation_space.bounds
        return spaces.Box(
            np.concatenate([task_ub, env_ub]),
            np.concatenate([task_lb, env_lb]))

    @overrides
    def step(self, action):
        obs, reward, done, info = super(OneHotMultiTaskEnv, self).step(action)
        oh_obs = self._obs_with_one_hot(obs)
        return Step(oh_obs, reward, done, **info)

    def _obs_with_one_hot(self, obs):
        oh_obs = np.concatenate([self.active_task_one_hot, obs])
        return oh_obs
