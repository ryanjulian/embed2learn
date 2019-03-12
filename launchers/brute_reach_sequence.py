import os.path as osp

from garage.config import LOG_DIR
import joblib
import numpy as np
import tensorflow as tf
from sawyer.mujoco import SimpleReacherEnv
from scipy.optimize import brute

from embed2learn.envs import DiscreteEmbeddedPolicyEnv


# USE_LOG = "local/sawyer-reach-embed-8goal/sawyer_reach_embed_8goal_2018_08_19_17_09_21_0001/"
# USE_LOG = "local/sawyer-reach-embed-tanh/sawyer_reach_embed_tanh_2018_08_22_09_45_54_0001/"
# latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_485.pkl")

USE_LOG = "local/sawyer-reach-embed-notanh/sawyer_reach_embed_notanh_2018_08_23_12_38_13_0001"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_300.pkl")

TASK_GOALS = [
  # (  z     x,    y)
    (0.5,  0.3, 0.15),
    (0.5, -0.3, 0.15),
    (0.5,  0.3,  0.3),
    (0.5, -0.3,  0.3),
    (0.7,  0.3, 0.15),
    (0.7, -0.3, 0.15),
    (0.7,  0.3,  0.3),
    (0.7, -0.3,  0.3),
]

GOAL = (0.7, 0., 0.15)

PATH_LENGTH = 20  # 80
SKIP_STEPS = 5  # 20

# for 80/20: [5, 2, 1, 4,]
# for 20/5: [7, 5, 7, 2]

def main():
    sess = tf.Session()
    sess.__enter__()

    snapshot = joblib.load(latent_policy_pkl)
    latent_policy = snapshot["policy"]
    ntasks = latent_policy.task_space.shape[0]
    tasks = np.eye(ntasks)
    latents = [latent_policy.get_latent(tasks[t])[1]["mean"] for t in range(ntasks)]
    print("Latents:\n\t", "\n\t".join(map(str, latents)))

    ITERATIONS = PATH_LENGTH // SKIP_STEPS

    inner_env = SimpleReacherEnv(goal_position=GOAL,
                                 control_method="position_control",
                                 completion_bonus=2.,
                                 action_scale=0.04)

    env = DiscreteEmbeddedPolicyEnv(inner_env,
                                    latent_policy,
                                    latents=latents,
                                    skip_steps=SKIP_STEPS,
                                    deterministic=True)

    def f(x):
        env.reset()
        reward = 0.
        # first go to the desired embedding
        env.step(int(x[0]))
        for i in range(ITERATIONS):
            obs, r, done, info = env.step(int(x[i]))
            reward += r
        print(x, "\tr:", reward)
        return -reward  # optimizers minimize by default

    print("Brute-forcing", ntasks ** ITERATIONS, "combinations...")
    ranges = (slice(0, ntasks, 1),) * ITERATIONS
    result = brute(f, ranges, disp=True, finish=None)
    print("Result:", result)

    markers = [dict(
        pos=GOAL,
        size=0.01 * np.ones(3),
        label="Goal",
        rgba=np.array([1., 0.8, 0., 1.])
    )]
    for i, g in enumerate(TASK_GOALS):
        markers.append(dict(
            pos=g,
            size=0.01 * np.ones(3),
            label="Task {}".format(i + 1),
            rgba=np.array([1., 0.2, 0., 1.])
        ))
    while True:
        env.reset()
        reward = 0.
        # first go to the desired embedding
        env.step(int(result[0]))
        for i in range(ITERATIONS):
            obs, r, done, info = env.step(int(result[i]),
                                          animate=True,
                                          markers=markers)
            reward += r
        print(result, "\tr:", reward)


if __name__ == "__main__":
    main()
