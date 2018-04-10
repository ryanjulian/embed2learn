import random

from cached_property import cached_property
import numpy as np

from rllab.envs.normalized_env import NormalizedEnv
from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.base import Step

from sandbox.rocky.tf.envs.base import TfEnv as BaseTfEnv
from sandbox.rocky.tf.envs.base import to_tf_space


class MultiTaskEnv(Env):
    def __init__(self, task_env_cls=None, task_args=None, task_kwargs=None):
        Serializable.quick_init(self, locals())
        self._task_envs = [
            task_env_cls(*t_args, **t_kwargs)
            for t_args, t_kwargs in zip(task_args, task_kwargs)
        ]
        self._active_env = None
        self._choose_active_task()

    def reset(self, **kwargs):
        self._choose_active_task()
        return self._active_env.reset(**kwargs)

    @property
    def action_space(self):
        return self._task_envs[0].action_space

    @property
    def observation_space(self):
        return self._task_envs[0].observation_space

    def step(self, action):
        obs, reward, done, info = self._active_env.step(action)
        info['task'] = self.active_task_one_hot
        return Step(obs, reward, done, **info)

        return

    def render(self, *args, **kwargs):
        return self._active_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        return self._active_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._task_envs[0].horizon

    def terminate(self):
        for env in self._task_envs:
            env.terminate()

    def get_param_values(self):
        return self._active_env.get_param_values()

    def set_param_values(self, params):
        self._active_env.set_param_values(params)

    @property
    def task_space(self):
        n = len(self._task_envs)
        one_hot_ub = np.ones(n)
        one_hot_lb = np.zeros(n)
        return spaces.Box(one_hot_lb, one_hot_ub)

    @property
    def active_task(self):
        return self._task_envs.index(self._active_env)

    @property
    def active_task_one_hot(self):
        one_hot = np.zeros(self.task_space.shape)
        one_hot[self.active_task] = self.task_space.high[self.active_task]
        return one_hot

    @property
    def num_tasks(self):
        return len(self._task_envs)

    def _choose_active_task(self):
        # uniform random
        #return random.choice(self._task_envs)

        # round robin
        if self._active_env is None:
            self._active_env = self._task_envs[0]
        else:
            self._active_env = self._task_envs[(
                self.active_task + 1) % self.num_tasks]


class TfEnv(BaseTfEnv):
    @cached_property
    def task_space(self):
        return to_tf_space(self.wrapped_env.task_space)

    @property
    def active_task_one_hot(self):
        return self.wrapped_env.active_task_one_hot

    @property
    def active_task(self):
        return self.wrapped_env.active_task


class NormalizedMultiTaskEnv(NormalizedEnv):
    @property
    def task_space(self):
        return self._wrapped_env.task_space

    @property
    def active_task_one_hot(self):
        return self.wrapped_env.active_task_one_hot

    @property
    def active_task(self):
        return self.wrapped_env.active_task


normalize = NormalizedMultiTaskEnv
