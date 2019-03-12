import numpy as np

from garage.core.serializable import Serializable
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
from garage.misc.overrides import overrides

class SequenceReacherEnv(ReacherEnv):

    def __init__(self, sequence=None, subgoal_bonus=5., **kwargs):
        self._sequence = [np.array(subgoal) for subgoal in sequence]
        self._reached = 0
        self._n_goals = len(sequence)
        self._subgoal_bonus = subgoal_bonus
        # Disable success function to make sure things are correct
        def success_fn(env, achieved_goal, desired_goal, _info: dict):
            return False
        ReacherEnv.__init__(
            self,
            goal_position=self._sequence[self._reached],
            success_fn=success_fn,
            **kwargs)

    @overrides
    def get_obs(self):
        obs = super(SequenceReacherEnv, self).get_obs()
        observations = obs["observation"].copy()
        seq_state = np.zeros(self._n_goals)
        if self._reached > 0:
            seq_state[self._reached - 1] = 1.
        obs_with_sequence_state = np.concatenate([observations, seq_state])
        obs["observation"] = obs_with_sequence_state
        return obs

    @overrides
    def compute_reward(self, achieved_goal, desired_goal, info):
        current_goal = self._sequence[self._reached]
        d = np.linalg.norm(achieved_goal - current_goal, axis=-1)
        if d < self._distance_threshold:
            return self._subgoal_bonus
        return -d
    @overrides
    def step(self, action):
        obs, _, _, info = super(SequenceReacherEnv, self).step(action)

        # Recalculate the reward here to make sure things are right
        # Need some cleanup later...
        reward = self.compute_reward(self.gripper_position, None, None)

        # Success
        d = np.linalg.norm(self.gripper_position - self._sequence[self._reached], axis=-1)
        is_success = False

        done = False
        if d < self._distance_threshold:
            self._reached += 1
            if self._reached == self._n_goals:
                reward = self._completion_bonus
                done = True
            else:
                reward = self._subgoal_bonus
                self._goal = self._sequence[self._reached]

        info["n_reached_goal"] = self._reached

        return obs, reward, done, info

    def reset(self):
        obs = super(SequenceReacherEnv, self).reset()
        self._reached = 0
        self._goal = self._sequence[0]
        return obs

    def render(self, mode="human"):
        viewer = self.get_viewer()
        for i, g in enumerate(self._sequence):
            viewer.add_marker(pos=g, label="task_{}".format(i+1), size=0.01*np.ones(3),)

        super(SequenceReacherEnv, self).render(mode=mode)

    def get_state(self):
        state = np.concatenate([[self._reached], self.joint_positions[2:]])
        return state

    def set_state(self, state):
        self._reached = int(state[0])
        self._goal = self._sequence[self._reached]
        super(SequenceReacherEnv, self).set_state(state[1:])


class SimpleReacherSequenceEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(SequenceReacherEnv(*args, **kwargs))
