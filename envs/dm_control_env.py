import numpy as np

from dm_control import suite

from rllab.envs import Env
from rllab.spaces import Box
from rllab.envs import Step

from sandbox.embed2learn.envs import DmControlViewer

RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_CAMERA_ID = 0


class DmControlEnv(Env):
    def __init__(self, *args, **task_kwargs):
        self.dm_env = suite.load(*args, task_kwargs=task_kwargs)
        self.viewer = None

    @property
    def observation_space(self):
        # TODO(rjulian): use dm_control.rl.control
        obs = self.__observe()
        return Box(low=-np.inf, high=np.inf, shape=obs.shape)

    @property
    def action_space(self):
        #TODO(rjulian): use dm_control.rl.control
        action_spec = self.dm_env.action_spec()
        return Box(
            low=np.min(action_spec.minimum),
            high=np.max(action_spec.maximum),
            shape=action_spec.shape)

    def reset(self):
        self.dm_env.reset()
        return self.__observe()

    def step(self, action):
        step = self.dm_env.step(action)
        reward = step.reward
        if reward is None:
            reward = 0

        return Step(
            observation=self.__observe(), reward=reward, done=step.last())

    def render(self):
        # TODO(rjulian): real MjViewer support
        if not self.viewer:
            self.viewer = DmControlViewer()
        image = self.dm_env.physics.render(RENDER_HEIGHT, RENDER_WIDTH,
                                           RENDER_CAMERA_ID)
        self.viewer.loop_once(image)

    def __observe(self):
        dm_obs = self.dm_env.task.get_observation(self.dm_env.physics)
        return np.hstack(entry.flatten() for entry in dm_obs.values())
