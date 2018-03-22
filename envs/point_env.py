import numpy as np

from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step


class PointEnv(Env):
    def __init__(self, goal=(0, 0), *args, **kwargs):
        self._goal = np.array(goal, dtype=np.float32)
        super(PointEnv, self).__init__(*args, **kwargs)

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2, ))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2, ))

    def reset(self):
        start = np.random.uniform(-1, 1, size=(2, ))
        self._state = start + self._goal
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        goal_x, goal_y = self._goal
        reward = -((x - goal_x)**2 + (y - goal_y)**2)**0.5
        done = abs(x - goal_x) < 0.01 and abs(y - goal_y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)
