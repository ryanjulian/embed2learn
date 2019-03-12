"""
Model Predictive Controller
"""

import numpy as np


class MPCPolicy:

    def __init__(self, embedding, n_learned_skills, inner_env, inner_policy, gamma=1., n_sampled_action=50, rollout_length=3):
        self._embedding = embedding
        self._n_learned_skills = n_learned_skills
        self._inner_env = inner_env
        self._inner_policy = inner_policy
        self._n_sampled_action = n_sampled_action
        self._rollout_length = rollout_length
        self._gamma = gamma

    def get_action(self, observation, state):
        actions = self._sample_actions()
        max_reward = -np.inf
        best_action = None
        for i in range(len(actions)):
            a = actions[i]
            r = self._try_action(a, state)
            if r >= max_reward or best_action is None:
                max_reward = r
                best_action = a.copy()
        return best_action

    def _try_action(self, action, state):
        self._inner_env.set_state(state)
        o = self._inner_env.env.get_obs()["observation"]

        reward = 0
        for i in range(self._rollout_length):
            a, agent_info = self._inner_policy.get_action_from_latent(action, o[:10])
            o, r, done, _ = self._inner_env.step(agent_info["mean"])
            reward += r * (self._gamma ** i)
            if done:
               break
        return reward

    def _sample_actions(self):
        one_hots = np.identity(self._n_learned_skills)
        latents, dist_info = self._embedding.get_latents(one_hots)

        results = []
        for i in range(self._n_sampled_action):
            idx = np.random.randint(low=0, high=self._n_learned_skills)
            mean, std = dist_info["mean"][idx][:], dist_info["log_std"][idx][:]
            noise = np.random.normal(size=latents.shape[1])
            results.append(mean + noise * std)

        return results
