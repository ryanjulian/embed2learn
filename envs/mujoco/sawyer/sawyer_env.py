from os import path as osp

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv

ACTION_LIMIT = 0.5


class SawyerEnv(MujocoEnv, Serializable):
    def __init__(self, target, *args, **kwargs):
        super(SawyerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        raise NotImplementedError

    @overrides
    @property
    def action_space(self):
        shape = self.model.actuator_ctrlrange[:, 0].shape
        lb = np.full(shape, -ACTION_LIMIT)
        ub = np.full(shape, ACTION_LIMIT)
        return spaces.Box(lb, ub)

    def step(self, action):
        raise NotImplementedError
