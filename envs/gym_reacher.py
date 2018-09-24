import gym
from gym.envs.mujoco.reacher import ReacherEnv
import numpy as np


class Reacher(ReacherEnv):
    """Add task setting option to the gym 2 DoF Reacher for multitask training"""

    def __init__(self, target=None):
        ReacherEnv.__init__(self)
        self.goal = target

    def reset_model(self):
        # Remove all the randomness..
        qpos = self.init_qpos
        qpos[-2:] = self.goal[:]
        qvel = self.init_qvel
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
