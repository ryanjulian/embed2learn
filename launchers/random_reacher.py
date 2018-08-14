import ipdb
import numpy as np
import time

from garage.envs.mujoco.sawyer import SimpleReacherEnv

env = SimpleReacherEnv(
    goal_position=(0.4, -0.3, 0.15),
    control_method="position_control",
    # control_cost_coeff=1.0,
    # action_scale=0.04,
    randomize_start_jpos=True,
    completion_bonus=0.0,
    # terminate_on_collision=True,
    # collision_penalty=1.,
)

low = np.array(
            [-3.0503, -3.8095, -3.0426, -3.0439, -2.9761, -2.9761, -4.7124])
high = np.array(
    [3.0503, 2.2736, 3.0426, 3.0439, 2.9761, 2.9761, 4.7124])

while True:

    k = 10000
    errors = []
    for _ in range(20):
        start_obs = env.reset()
        pos = start_obs[:7]
        for _ in range(500):
            next_step = env.action_space.sample()
            next_pos = pos + env.action_space.sample()
            next_pos_clip = np.clip(next_pos, low, high)

            if not all(np.isclose(next_pos, next_pos_clip)):
                break

            obs, r, done, info = env.step(next_step)
            pos = obs[:7]
            errors.append(pos - next_pos)
            env.render()

    errors = np.array(errors)
    bias = np.mean(errors, axis=0)

    ipdb.set_trace()
