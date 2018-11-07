import random

from cached_property import cached_property
import gym
import numpy as np

from garage import spaces
from garage.core import Parameterized
from garage.core import Serializable
from garage.envs import Step
from garage.envs.normalized_env import NormalizedEnv

from garage.tf.envs import TfEnv as BaseTfEnv


def round_robin(num_tasks, last_task):
    if last_task is None:
        return 0

    return (last_task + 1) % num_tasks


def uniform_random(num_tasks, last_task):
    return random.randint(0, num_tasks)


class MultiTaskEnv(gym.Env, Parameterized):
    def __init__(self,
                 task_selection_strategy=round_robin,
                 task_env_cls=None,
                 task_args=None,
                 task_kwargs=None):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._task_envs = [
            task_env_cls(*t_args, **t_kwargs)
            for t_args, t_kwargs in zip(task_args, task_kwargs)
        ]
        self._task_selection_strategy = task_selection_strategy
        self._active_task = None

    def reset(self, **kwargs):
        self._active_task = self._task_selection_strategy(
            self.num_tasks, self.active_task)
        return self.active_env.reset(**kwargs)

    @property
    def action_space(self):
        return self.active_env.action_space

    @property
    def observation_space(self):
        return self.active_env.observation_space

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        info['task'] = self.active_task_one_hot
        return Step(obs, reward, done, **info)

    def render(self, *args, **kwargs):
        return self.active_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        return self.active_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self.active_env.horizon

    def close(self):
        for env in self._task_envs:
            env.close()

    def get_params_internal(self, **tags):
        # return self.active_env.get_params_internal(**tags)
        return []

    # def set_param_values(self, params):
    #     self._active_env.set_param_values(params)

    @property
    def task_space(self):
        n = len(self._task_envs)
        one_hot_ub = np.ones(n)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.float32)

    @property
    def active_task(self):
        return self._active_task

    @property
    def active_task_one_hot(self):
        one_hot = np.zeros(self.task_space.shape)
        t = self.active_task or 0
        one_hot[t] = self.task_space.high[t]
        return one_hot

    @property
    def active_env(self):
        return self._task_envs[self.active_task or 0]

    @property
    def num_tasks(self):
        return len(self._task_envs)

    @property
    def task_selection_strategy(self):
        return self._task_selection_strategy

    @task_selection_strategy.setter
    def task_selection_strategy(self, strategy):
        self._task_selection_strategy = strategy


class TfEnv(BaseTfEnv):
    @cached_property
    def task_space(self):
        return self._to_garage_space(self.env.task_space)

    @property
    def active_task_one_hot(self):
        return self.env.active_task_one_hot

    @property
    def active_task(self):
        return self.env.active_task


class NormalizedMultiTaskEnv(NormalizedEnv, Parameterized):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        NormalizedEnv.__init__(self, env)

    @property
    def task_space(self):
        return self.env.task_space

    @property
    def active_task_one_hot(self):
        return self.env.active_task_one_hot

    @property
    def active_task(self):
        return self.env.active_task

    def get_params_internal(self, *args, **kwargs):
        return self.env.get_params_internal(*args, **kwargs)


normalize = NormalizedMultiTaskEnv
