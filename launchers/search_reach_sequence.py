import copy
import pickle
from queue import PriorityQueue
import os.path as osp

from garage.core import Serializable
from garage.envs.mujoco.sawyer import ReacherEnv
from garage.envs.mujoco.sawyer.sawyer_env import SawyerEnvWrapper
import joblib
import numpy as np
from scipy.optimize import brute
import tensorflow as tf
from tqdm import tqdm

from embed2learn.envs.discrete_embedded_policy_env import DiscreteEmbeddedPolicyEnv


USE_LOG = "sawyer-reach-embed-notanh/sawyer_reach_embed_notanh_2018_08_23_12_38_13_0001"
# USE_LOG = "push_embed/sawyer_pusher_rel_obs_embed_udlr_2018_08_23_15_32_40_0001"
LOG_DIR = "/home/eric/.deep-rl-docker/garage_embed/data/local"
latent_policy_pkl = osp.join(LOG_DIR, USE_LOG, "itr_315.pkl")

GOAL_SEQUENCE = [
    (0.6, 0.3, 0.15),
    (0.6, -0.3, 0.15),
    (0.6, 0., 0.3),
    (0.6, 0.3, 0.15),
]

PATH_LENGTH = 140  # 80
SKIP_STEPS = 5  # 20

SAVE_N_ROLLOUTS = 20

SEARCH_METHOD = "ucs"  # "greedy"  # "brute"

ITERATIONS = PATH_LENGTH // SKIP_STEPS


class SequenceReacherEnv(ReacherEnv):
    def __init__(self, sequence=None, **kwargs):
        self._sequence = [np.array(subgoal) for subgoal in sequence]
        self._reached = 0
        self._n_goals = len(sequence)
        ReacherEnv.__init__(
            self,
            goal_position=self._sequence[self._reached],
            **kwargs)

    def step(self, action):
        obs, _, _, info = super(SequenceReacherEnv, self).step(action)

        current_goal = self._sequence[self._reached]
        d = np.linalg.norm(self.gripper_position - current_goal, axis=-1)
        reward = -d

        done = self._reached == len(self._sequence) - 1
        if d < self._distance_threshold and self._reached < len(self._sequence) - 1:
            self._reached += 1
            self._goal = self._sequence[self._reached]
        info["n_reached_goal"] = self._reached

        reward += self._reached
        return obs, reward, done, info

    def reset(self):
        obs = super(SequenceReacherEnv, self).reset()
        self._reached = 0
        self._goal = self._sequence[0]
        return obs


class SimpleSequenceReacherEnv(SawyerEnvWrapper, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self.reward_range = None
        self.metadata = None
        super().__init__(SequenceReacherEnv(*args, **kwargs))


def greedy(env: DiscreteEmbeddedPolicyEnv, ntasks: int):
    sequence = []
    for l in range(ITERATIONS):
        best_r, best_a = -np.inf, 0
        for a in range(ntasks):
            env.set_sequence(sequence)
            obs, r, done, info = env.step(a)
            if r > best_r:
                best_r, best_a = r, a
        sequence.append(best_a)
    return sequence


def ucs(env: DiscreteEmbeddedPolicyEnv, ntasks: int):
    queue = PriorityQueue()
    for a in range(ntasks):
        r = env.set_sequence([a])
        queue.put((-r, [a]))
    while not queue.empty():
        curr_r, curr_s = queue.get()
        print("%.4f\t" % curr_r, curr_s)
        if len(curr_s) == ITERATIONS:
            return curr_s
        for a in range(ntasks):
            seq = curr_s + [a]
            r = env.set_sequence(seq)
            queue.put((-r, curr_s + [a]))
    return []


TASKS = [
    # (  z     x,    y)
    (0.5, 0.3, 0.15),
    (0.5, -0.3, 0.15),
    (0.5, 0.3, 0.3),
    (0.5, -0.3, 0.3),
    (0.7, 0.3, 0.15),
    (0.7, -0.3, 0.15),
    (0.7, 0.3, 0.3),
    (0.7, -0.3, 0.3),
]


def main():
    sess = tf.Session()
    sess.__enter__()

    snapshot = joblib.load(latent_policy_pkl)
    latent_policy = snapshot["policy"]
    ntasks = latent_policy.task_space.shape[0]
    tasks = np.eye(ntasks)
    latents = [latent_policy.get_latent(tasks[t])[1]["mean"] for t in range(ntasks)]
    print("Latents:\n\t", "\n\t".join(map(str, latents)))

    inner_env = SimpleSequenceReacherEnv(sequence=GOAL_SEQUENCE,
                                         control_method="position_control",
                                         completion_bonus=0.,
                                         randomize_start_jpos=False,
                                         action_scale=0.04,
                                         distance_threshold=0.05)

    env = DiscreteEmbeddedPolicyEnv(inner_env,
                                    latent_policy,
                                    latents=latents,
                                    skip_steps=SKIP_STEPS,
                                    deterministic=True)

    if SEARCH_METHOD == "brute":
        def f(x):
            env.reset()
            reward = 0.
            for i in range(ITERATIONS):
                obs, r, done, info = env.step(int(x[i]))
                reward += r
            print(x, "\tr:", reward)
            return -reward  # optimizers minimize by default

        print("Brute-forcing", ntasks ** ITERATIONS, "combinations...")
        ranges = (slice(0, ntasks, 1),) * ITERATIONS
        result = brute(f, ranges, disp=True, finish=None)
    elif SEARCH_METHOD == "greedy":
        result = greedy(env, ntasks)
    elif SEARCH_METHOD == "ucs":
        result = ucs(env, ntasks)
    else:
        raise NotImplementedError

    print("Result:", result)

    markers = []
    for i, g in enumerate(GOAL_SEQUENCE):
        markers.append(dict(
            pos=g,
            size=0.01 * np.ones(3),
            label="Goal {}".format(i + 1),
            rgba=np.array([1., 0.8, 0., 1.])
        ))
    for i, g in enumerate(TASKS):
        markers.append(dict(
            pos=g,
            size=0.01 * np.ones(3),
            label="Task {}".format(i + 1),
            rgba=np.array([0.5, 0.5, 0.5, 0.8])
        ))

    print("Collecting %i rollouts..." % SAVE_N_ROLLOUTS)
    rollouts = []
    for _ in tqdm(range(SAVE_N_ROLLOUTS)):
        env.reset()
        reward = 0.
        seq_info = {
            "latents": [],
            "latent_indices": [],
            "observations": [],
            "actions": [],
            "infos": [],
            "rewards": [],
            "dones": []
        }
        for i in range(ITERATIONS):
            obs, r, done, info = env.step(int(result[i]))
            seq_info["latents"] += info["latents"]
            seq_info["latent_indices"] += info["latent_indices"]
            seq_info["observations"] += info["observations"]
            seq_info["actions"] += info["actions"]
            seq_info["infos"] += info["infos"]
            seq_info["rewards"] += info["rewards"]
            seq_info["dones"] += info["dones"]
            reward += r
        # print(result, "\tr:", reward)
        rollouts.append(copy.deepcopy(seq_info))

    pickle.dump(rollouts, open("rollout_search_sequencer.pkl", "wb"))

    while True:
        env.reset()
        reward = 0.
        for i in range(ITERATIONS):
            obs, r, done, info = env.step(int(result[i]),
                                          animate=True,
                                          markers=markers)
            reward = r
        print(result, "\tr:", reward)


if __name__ == "__main__":
    main()
