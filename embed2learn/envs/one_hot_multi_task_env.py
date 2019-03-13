import akro
from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
import numpy as np

from embed2learn.envs import MultiTaskEnv


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
        return akro.Box(
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
