from collections import deque

import numpy as np
import pygame

from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.misc.overrides import overrides

from rllab.spaces.box import Box

MAX_SHOWN_TRACES = 10

TRACE_COLORS = [(80, 150, 0), (100, 180, 10), (100, 210, 30), (140, 230, 50),
                (180, 250, 150)]
BRIGHT_COLOR = (200, 200, 200)
DARK_COLOR = (150, 150, 150)


class PointEnv(Env, Serializable):

    def __init__(self, *args, goal=(0, 0), show_traces=True, **kwargs):
        super(PointEnv, self).__init__(*args, **kwargs)

        self._goal = np.array(goal, dtype=np.float32)
        self._point = np.zeros(2, dtype=np.float32)
        self._start = np.zeros(2, dtype=np.float32)

        self.screen = None
        self.screen_width = 500
        self.screen_height = 500
        self.zoom = 50.
        self.show_traces = show_traces

        self._traces = deque(maxlen=MAX_SHOWN_TRACES)

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2, ))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2, ))

    def reset(self):
        self._point = np.copy(self._start)
        observation = np.copy(self._point)
        self._traces.append([])
        return observation

    def step(self, action):
        self._point = self._point + action
        x, y = self._point
        self._traces[-1].append((x, y))
        reward = -np.linalg.norm(self._point - self._goal)
        done = np.linalg.norm(self._point - self._goal, ord=np.inf) < 0.1
        next_observation = np.copy(self._point)
        return Step(observation=next_observation, reward=reward, done=done)

    def _to_screen(self, position):
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
        pygame.draw.circle(self.screen, (0, 0, 255),
                           self._to_screen(self._start), 10, 0)

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

    def terminate(self):
        if self.screen:
            pygame.quit()
