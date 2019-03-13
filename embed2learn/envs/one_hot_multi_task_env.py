from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
import gym
import numpy as np

from embed2learn.envs import MultiTaskEnv


class OneHotMultiTaskEnv(MultiTaskEnv, Serializable):
    @overrides
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return self._obs_with_one_hot(obs)

    @property
    @overrides
    def observation_space(self):
        task_ub, task_lb = self.task_space.low, self.task_space.high
        env_os = super().observation_space
        env_ub, env_lb = env_os.low, env_os.high
        return gym.spaces.Box(
            np.concatenate([task_ub, env_ub]),
            np.concatenate([task_lb, env_lb]), dtype=np.float32)

    @overrides
    def step(self, action):
        obs, reward, done, info = super().step(action)
        oh_obs = self._obs_with_one_hot(obs)
        return Step(oh_obs, reward, done, **info)

    def _obs_with_one_hot(self, obs):
        oh_obs = np.concatenate([self.active_task_one_hot, obs])
        return oh_obs
