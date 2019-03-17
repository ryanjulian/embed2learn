from collections import deque

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
import gym
import numpy as np
import pygame

from embed2learn.envs.util import colormap

MAX_SHOWN_TRACES = 10
TRACE_COLORS = colormap(MAX_SHOWN_TRACES)
BRIGHT_COLOR = (200, 200, 200)
DARK_COLOR = (150, 150, 150)


class PointEnv(gym.Env, Serializable):
    def __init__(
            self,
            goal=(1, 1),
            random_start=False,
            show_traces=True,
            completion_bonus=0.,
            never_done=False,
            action_scale=1.,
        ):
        Serializable.quick_init(self, locals())
        # Parameterized.__init__(self)

        self._goal = np.array(goal, dtype=np.float32)
        self._point = np.zeros(2)
        self._completion_bonus = completion_bonus
        self._never_done = never_done
        self._action_scale = action_scale

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
            self._point = np.random.uniform(low=-4, high=4, size=self._point.shape)
        else:
            self._point = np.zeros_like(self._goal)
        self._traces.append([tuple(self._point)])
        return np.copy(self._point)

    def step(self, action):
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a *= self._action_scale
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._point = np.clip(self._point + a, -5, 5)
        self._traces[-1].append(tuple(self._point))

        dist = np.linalg.norm(self._point - self._goal)
        done = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist
        is_success = False
        # completion bonus
        if done:
            is_success = True
            reward += self._completion_bonus

        # sometimes we don't want to terminate
        done = done and not self._never_done

        return Step(np.copy(self._point), reward, done, is_success=is_success)

    def _to_screen(self, position):
        position = np.nan_to_num(position)
        return (int(self.screen_width / 2 + position[0] * self.zoom),
                int(self.screen_height / 2 - position[1] * self.zoom))

    @overrides
    def render(self, mode="human"):

        if self.screen is None and mode == "human":
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

        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            pass

    def log_diagnostics(self, paths):
        pass

    def close(self):
        if self.screen:
            pygame.quit()
