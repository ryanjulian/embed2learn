import numpy as np

import pygame
from Box2D import b2Color

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces.box import Box
from rllab.envs.base import Step

MAX_SHOWN_TRACES = 10

TRACE_COLORS = [
    (80, 150, 0),
    (100, 180, 10),
    (100, 210, 30),
    (140, 230, 50),
    (180, 250, 150)
]


class PointEnv(Env, Serializable):

    def __init__(self, goal=(0, 0), show_traces=True, *args, **kwargs):
        super(PointEnv, self).__init__(*args, **kwargs)

        self._goal = np.array(goal, dtype=np.float32)
        self._point = np.zeros(2)

        self.screen = None
        self.screen_width = 500
        self.screen_height = 500
        self.zoom = 50.
        self.show_traces = show_traces

        self.traces = [[]]

        Serializable.__init__(self, *args, **kwargs)

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2, ))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2, ))

    def reset(self):
        start = np.random.uniform(-1, 1, size=(2, ))
        self._point = start + self._goal
        observation = np.copy(self._point)
        if self.show_traces:
            self.traces.append([])
            if len(self.traces) > MAX_SHOWN_TRACES:
                self.traces = self.traces[-MAX_SHOWN_TRACES:]
        return observation

    def step(self, action):
        self._point = self._point + action
        x, y = self._point
        if self.show_traces:
            self.traces[-1].append((x, y))
        goal_x, goal_y = self._goal
        reward = -((x - goal_x)**2 + (y - goal_y)**2)**0.5
        done = abs(x - goal_x) < 0.01 and abs(y - goal_y) < 0.01
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
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        self.screen.fill((255, 255, 255))

        # draw grid
        bright = b2Color(0.8, 0.8, 0.8).bytes
        dark = b2Color(0.6, 0.6, 0.6).bytes
        for x in range(25):
            dx = -6. + x * 0.5
            pygame.draw.line(self.screen, dark if x % 2 == 0 else bright,
                             self._to_screen((dx, -10)), self._to_screen((dx, 10)))
        for y in range(25):
            dy = -6. + y * 0.5
            pygame.draw.line(self.screen, dark if y % 2 == 0 else bright,
                             self._to_screen((-10, dy)), self._to_screen((10, dy)))

        # draw point
        pygame.draw.circle(
            self.screen,
            b2Color(0.2, 0.8, 0.).bytes,
            self._to_screen(self._point), 10, 0)

        # draw goal
        pygame.draw.circle(
            self.screen,
            b2Color(1.0, 0.2, 0.).bytes,
            self._to_screen(self._goal), 10, 0)

        # draw traces
        if self.show_traces:
            for i, trace in self.traces:
                if len(trace) > 1:
                    pygame.draw.lines(self.screen,
                                      TRACE_COLORS[-min(len(TRACE_COLORS)-1, i)],
                                      False,
                                      [self._to_screen(p) for p in trace])

        pygame.display.flip()
