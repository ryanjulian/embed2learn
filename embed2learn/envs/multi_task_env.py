import random

from cached_property import cached_property
from garage.core import Serializable
from garage.envs.normalized_env import NormalizedEnv
from garage.envs import Step
from garage.tf.envs import TfEnv as BaseTfEnv
import gym
import numpy as np


def round_robin(num_tasks, last_task):
    if last_task is None:
        return 0

    return (last_task + 1) % num_tasks


def uniform_random(num_tasks, last_task):
    return random.randint(0, num_tasks)


class MultiTaskEnv(gym.Env, Serializable):
    def __init__(self,
                 task_selection_strategy=round_robin,
                 task_env_cls=None,
                 task_args=None,
                 task_kwargs=None):
        Serializable.quick_init(self, locals())

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

    def close(self):
        for env in self._task_envs:
            env.close()

    @property
    def task_space(self):
        t = self.active_task or 0
        n = self._task_envs[t]._max_sentence_length
        one_hot_ub = np.ones(n) * (self._task_envs[t]._sentence_code_dim - 1)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.int32)
        # return {
        #     'sentence_code_dim': self._task_envs[t]._sentence_code_dim,
        #     'max_sentence_length': self._task_envs[t]._max_sentence_length
        # }

    @property
    def active_task(self):
        return self._active_task

    @property
    def active_task_one_hot_gt(self):
        one_hot = np.zeros(len(self._task_envs))
        t = self.active_task or 0
        one_hot[t] = 1
        return one_hot

    @property
    def active_task_one_hot(self):
        t = self.active_task or 0
        goal_description = self._task_envs[t]._goal_description
        return goal_description
        # return np.array(self._task_envs[t]._goal_description, np.int32)

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
        return self._to_akro_space(self.env.task_space)

    @property
    def active_task_one_hot(self):
        return self.env.active_task_one_hot
    
    @property
    def active_task_one_hot_gt(self):
        return self.env.active_task_one_hot_gt

    @property
    def active_task(self):
        return self.env.active_task


class NormalizedMultiTaskEnv(NormalizedEnv, Serializable):
    def __init__(self, env):
        super().__init__(self, env)
        Serializable.quick_init(self, locals())

    @property
    def task_space(self):
        return self.env.task_space

    @property
    def active_task_one_hot(self):
        return self.env.active_task_one_hot

    @property
    def active_task_one_hot_gt(self):
        return self.env.active_task_one_hot_gt

    @property
    def active_task(self):
        return self.env.active_task


normalize = NormalizedMultiTaskEnv
