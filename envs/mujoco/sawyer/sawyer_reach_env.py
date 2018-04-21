from os import path as osp

from sandbox.embed2learn.envs.mujoco.sawyer.sawyer_env import SawyerEnv

from sandbox.embed2learn.envs.mujoco.utils import mujoco_model_path
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.mujoco_py import MjViewer
from rllab.envs.base import Step

import numpy as np

FILE = osp.join('sawyer', 'reach.xml')

DEFAULT_TARGET = 'target0'


class SawyerReachEnv(SawyerEnv, Serializable):
    def __init__(self, target=DEFAULT_TARGET, *args, **kwargs):
        self.target = target

        kwargs['file_path'] = mujoco_model_path(FILE)
        super(SawyerReachEnv, self).__init__(self.target, *args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return np.concatenate([self.joint_angles(), self.finger_to_target()])

    @overrides
    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(
                title='Simulate: target = {}'.format(self.target))
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

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
        return self._get_geom_pos('leftclaw_it') - self._get_geom_pos(
            self.target)

    def finger_to_target_dist(self):
        return np.linalg.norm(self.finger_to_target())

    def target_size(self):
        return self._get_geom_size(self.target)

    def _get_geom_pos(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.data.geom_xpos[idx]

    def _get_geom_size(self, geom_name):
        idx = self.model.geom_names.index(geom_name)
        return self.model.geom_size[idx][0]
