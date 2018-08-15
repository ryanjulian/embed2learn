from garage.misc.overrides import overrides
from garage.core import Serializable

from sandbox.embed2learn.envs import PointEnv


class SequencePointEnv(PointEnv):

    def __init__(self, goal_sequence=None, completion_bonus=0., action_scale=1.,):

        Serializable.quick_init(self, locals())
        sequence = [
            (0, 3),
            (3, 0),
            (0, -3),
            (-3, 0),
        ]
        self._goal_sequence = sequence if goal_sequence is None else goal_sequence

        super(SequencePointEnv, self).__init__(
            goal=self._goal_sequence[0], 
            random_start=False,
            never_done=False,
            completion_bonus=completion_bonus,
            action_scale=action_scale,
        )
        self._n_goals = len(self._goal_sequence)
        self._reached_goal = 0

    @overrides
    def step(self, action):

        # Step the action with point env
        obs, reward, done, info = super(SequencePointEnv, self).step(action)
        reward = 0
        # Recalculate the reward:
        if info["is_success"]:
            # Reached the current goal
            self._reached_goal += 1
            if self._reached_goal >= self._n_goals:
                done = True
            else:
                reward = 1
                done = False
                info["is_success"] = False
                self._goal = self._goal_sequence[self._reached_goal]

        reward += self._completion_bonus if info["is_success"] else 0

        return obs, reward, done, info

    @overrides
    def reset(self):
        obs = super(SequencePointEnv, self).reset()
        self._reached_goal = 0
        self._goal = self._goal_sequence[self._reached_goal]
        return obs
