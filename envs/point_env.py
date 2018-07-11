from collections import deque

import gym
import numpy as np
import pygame

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
from garage.spaces import Box

from sandbox.embed2learn.envs.util import colormap

MAX_SHOWN_TRACES = 10
# TRACE_COLORS = [
#     (80, 150, 0),
#     (100, 180, 10),
#     (100, 210, 30),
#     (140, 230, 50),
#     (180, 250, 150)
# ]  # yapf: disable
TRACE_COLORS = colormap(MAX_SHOWN_TRACES)
BRIGHT_COLOR = (200, 200, 200)
DARK_COLOR = (150, 150, 150)


class PointEnv(gym.Env, Parameterized):
    def __init__(self, goal=(1, 1), random_start=False, show_traces=True):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._goal = np.array(goal, dtype=np.float32)
        self._point = np.zeros(2)

        self.screen = None
        self.screen_width = 500
        self.screen_height = 500
        self.zoom = 50.
        self.show_traces = show_traces
        self.random_start = random_start

        self._traces = deque(maxlen=MAX_SHOWN_TRACES)

    def get_params_internal(self, **tags):
        return []

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-0.1, high=0.1, shape=(2, ), dtype=np.float32)

    def reset(self):
        if self.random_start:
            self._point = np.random.uniform(size=self._point.shape)
        else:
            self._point = np.zeros_like(self._goal)
        self._traces.append([tuple(self._point)])
        return np.copy(self._point)

    def step(self, action):
        # enforce action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._point = self._point + action
        self._traces[-1].append(tuple(self._point))

        dist = np.linalg.norm(self._point - self._goal)
        done = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        #reward = -np.exp(dist)
        reward = -dist

        # alive penalty
        # reward -= 10.0

        # completion bonus
        if done:
            reward = 100

        # reward scaling
        # reward *= 10.0

        return Step(observation=np.copy(self._point), reward=reward, done=done)

    def _to_screen(self, position):
        position = np.nan_to_num(position)
        return (int(self.screen_width / 2 + position[0] * self.zoom),
                int(self.screen_height / 2 - position[1] * self.zoom))

    @overrides
    def render(self, **kwargs):

        if self.screen is None:
            pygame.init()
            caption = "Point Environment"
            pygame.display.set_caption(caption)
            self.screen = pygame.display.set_mode((self.screen_width,
                                                   self.screen_height))

        self.screen.fill((255, 255, 255))

        # draw grid
        for x in range(25):
            dx = -6. + x * 0.5
            pygame.draw.line(self.screen, DARK_COLOR
                             if x % 2 == 0 else BRIGHT_COLOR,
                             self._to_screen((dx, -10)),
                             self._to_screen((dx, 10)))
        for y in range(25):
            dy = -6. + y * 0.5
            pygame.draw.line(self.screen, DARK_COLOR
                             if y % 2 == 0 else BRIGHT_COLOR,
                             self._to_screen((-10, dy)),
                             self._to_screen((10, dy)))

        # draw starting point
        pygame.draw.circle(self.screen, (0, 0, 255), self._to_screen((0, 0)),
                           10, 0)

        # draw goal
        pygame.draw.circle(self.screen, (255, 40, 0),
                           self._to_screen(self._goal), 10, 0)

        # draw point
        pygame.draw.circle(self.screen, (40, 180, 10),
                           self._to_screen(self._point), 10, 0)

        # draw traces
        if self.show_traces:
            for i, trace in enumerate(self._traces):
                if len(trace) > 1:
                    pygame.draw.lines(
                        self.screen,
                        TRACE_COLORS[-min(len(TRACE_COLORS) - 1, i)], False,
                        [self._to_screen(p) for p in trace])

        pygame.display.flip()

    def log_diagnostics(self, paths):
        pass

    def close(self):
        if self.screen:
            pygame.quit()
