from os import path

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides

from sandbox.embed2learn.envs.mujoco.utils import mujoco_model_path

FILE = path.join('pr2', 'pr2_arm.xml')

ACTION_LIMIT = 0.25


class PR2ArmEnv(MujocoEnv, Serializable):
    def __init__(self, *args, **kwargs):
        kwargs['file_path'] = mujoco_model_path(FILE)
        super(PR2ArmEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self.joint_angles(),
            self.finger_to_target(),
        ])

    @overrides
    @property
    def action_space(self):
        shape = self.model.actuator_ctrlrange[:, 0].shape
        lb = np.full(shape, -ACTION_LIMIT)
        ub = np.full(shape, ACTION_LIMIT)
        return spaces.Box(lb, ub)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        distance_to_go = self.finger_to_target_dist()
        vel_cost = 1e-2 * np.linalg.norm(self.joint_velocities())
        reward = -distance_to_go - vel_cost

        done = self.finger_to_target_dist() < self.target_size()

        return Step(next_obs, reward, done)

    def joint_angles(self):
        return self.model.data.qpos.flat

    def joint_velocities(self):
        return self.model.data.qvel.flat

    def finger_to_target(self):
        return self._get_geom_pos('finger') - self._get_geom_pos('target')

    def finger_to_target_dist(self):
        return np.linalg.norm(self.finger_to_target())

    def target_size(self):
        return self._get_geom_size('target')

    def _get_geom_pos(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.data.geom_xpos[idx]

    def _get_geom_size(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.geom_size[idx][0]
