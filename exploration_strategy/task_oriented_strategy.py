import numpy as np


class TaskStrategy:

    def __init__(self, policy, multi_task_policy, prob=0.8, std=2, **kwargs):

        """
        @param simulator: a simulator of the environment
        @param sample_szie: the batch size of sample z
        """
        self.policy = policy
        self.multi_task_policy = multi_task_policy
        self.n_task = multi_task_policy.task_space.flat_dim
        self.embedding = multi_task_policy._embedding
        self.prob = prob
        self.std = std

    def get_action(self, itr, observation, policy):
        # TODO: add epoch length to do this
        self.prob -= 1e-5
        if np.random.uniform() <= self.prob:
            # Sampled from learned skill
            one_hot = np.zeros(shape=(self.n_task,))
            idx = np.random.randint(0, self.n_task)
            one_hot[idx] = 1
            mean, _ = self.embedding.get_latent(one_hot)
            z = mean + np.random.normal() * self.std
            info = dict(mean=mean, log_std=np.full_like(mean, np.log(self.std)))
        else:
            z, info = self.policy.get_action(observation)
        return z # , info

    def get_actions(self, itr, observations,  policy):
        zs = []
        means = []
        log_stds = []
        for i in range(len(observations)):
            z, info = self.get_action(observations[i])
            zs.append(z)
            # means.append(info['mean'])
            # log_stds.append(info['log_std'])
        return np.array(zs)  # , dict(mean=np.array(means), log_std=np.array(log_stds))

    def reset(self):
        pass