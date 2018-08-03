from garage.core import Serializable
from garage.core import Parameterized

from multiworld.envs.mujoco.sawyer_reach_torque.sawyer_reach_torque_env import SawyerReachTorqueEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

class FlatXYZReacher(FlatGoalEnv, Parameterized):

    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        super().__init__(SawyerReachXYZEnv(*args, **kwargs))


class FlatTorqueReacher(FlatGoalEnv, Parameterized):

    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        super().__init__(SawyerReachTorqueEnv(*args, **kwargs))
