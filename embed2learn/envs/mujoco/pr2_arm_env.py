from os import path

import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides

from embed2learn.envs.mujoco.utils import mujoco_model_path

FILE = path.join('pr2','pr2_arm.xml')

ACTION_LIMIT = 0.25

class PR2ArmEnv(MujocoEnv, Serializable):

    def __init__(self, *args, **kwargs):
        kwargs['file_path'] = mujoco_model_path(FILE)
        super(PR2ArmEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([
            self._joint_angles(),
            self._finger_to_target(),
        ])

    @overrides
    @property
    def action_space(self):
        shape = self.model.actuator_ctrlrange[:,0].shape
        lb = np.full(shape, -ACTION_LIMIT)
        ub = np.full(shape, ACTION_LIMIT)
        return spaces.Box(lb, ub)        

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        distance_to_go = self._finger_to_target_dist()
        
        reward = -distance_to_go - ctrl_cost
        done = self._finger_to_target_dist() < 1e-6
        if done:
            print("done!")

        return Step(next_obs, reward, done)

    def _joint_angles(self):
        return self.model.data.qpos.flat[2:] # Skip 2 DoFs of the target object

    def _finger_to_target(self):
        return self._get_geom_pos('finger') - self._get_geom_pos('target')

    def _finger_to_target_dist(self):
        return np.linalg.norm(self._finger_to_target())

    def _target_size(self):
        return self._get_geom_size('target')

    def _get_geom_pos(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.geom_pos[idx]

    def _get_geom_size(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.geom_size[idx][0]