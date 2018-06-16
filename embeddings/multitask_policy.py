#from garage.tf.core import Parameterized
from garage.tf.policies import Policy
from garage.tf.policies import StochasticPolicy

from sandbox.embed2learn.embeddings import Embedding
from sandbox.embed2learn.embeddings import StochasticEmbedding
from sandbox.embed2learn.embeddings.utils import concat_spaces


class MultitaskPolicy(Policy):
    def __init__(self, env_spec, embedding, task_space):
        #Parameterized.__init__(self)
        self._env_spec = env_spec
        self._embedding = embedding
        self._task_space = task_space
        self._task_observation_space = concat_spaces(
            self._task_space, self._env_spec.observation_space)

    # Should be implemented by all policies

    def get_action_from_onehot(self, observation, onehot):
        raise NotImplementedError

    def get_actions_from_onehot(self, observations, onehots):
        raise NotImplementedError

    def get_action_from_latent(self, observation, latent):
        raise NotImplementedError

    def get_actions_from_latent(self, observations, latents):
        raise NotImplementedError

    def get_latent(self, onehot):
        return self._embedding.get_latent(onehot)

    def reset(self, dones=None):
        pass

    @property
    def vectorized(self):
        """
        Indicates whether the policy is vectorized. If True, it should implement get_actions(), and support resetting
        with multiple simultaneous states.
        """
        return False

    @property
    def embedding(self):
        return self._embedding

    @property
    def latent_space(self):
        return self._embedding.latent_space

    @property
    def embedding_spec(self):
        return self._embedding.embedding_spec

    @property
    def task_space(self):
        return self._task_space

    @property
    def observation_space(self):
        return self._env_spec.observation_space

    @property
    def task_observation_space(self):
        return self._task_observation_space

    @property
    def action_space(self):
        return self._env_spec.action_space

    @property
    def env_spec(self):
        return self._env_spec

    @property
    def recurrent(self):
        """
        Indicates whether the policy is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the policy's state when taking an action.
        :return:
        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """
        Return keys and shapes for the information related to the policy's state when taking an action.
        :return:
        """
        return list()

    def terminate(self):
        """
        Clean up operation
        """
        pass

    def split_observation(self, observation):
        """
        Splits up observation into task onehot and vanilla environment observation.
        :param observation: task onehot concatenated with vanilla environment observation
        :return: tuple (task onehot, vanilla environment observation)
        """
        return observation[:self.task_space.flat_dim], observation[
            self.task_space.flat_dim:]


class StochasticMultitaskPolicy(StochasticPolicy, MultitaskPolicy):
    def __init__(self, env_spec, embedding: StochasticEmbedding, task_space):
        super().__init__(env_spec, embedding, task_space)
        self._embedding = embedding

    @property
    def embedding_distribution(self):
        """
        :rtype Distribution
        """
        return self._embedding.distribution

    def embedding_dist_info_sym(self, obs_var, state_info_vars):
        """
        Return the symbolic distribution information about the embedding.
        :param obs_var: symbolic variable for observations
        :param state_info_vars: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        return self._embedding.dist_info_sym

    def embedding_dist_info(self, obs, state_infos):
        """
        Return the distribution information about the embedding.
        :param obs_var: observation values
        :param state_info_vars: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        return self._embedding.dist_info

    @property
    def action_distribution(self):
        """
        :rtype Distribution
        """
        raise NotImplementedError

    def action_dist_info_sym(self, obs_var, state_info_vars):
        """
        Return the symbolic distribution information about the actions.
        :param obs_var: symbolic variable for observations
        :param state_info_vars: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError

    def action_dist_info(self, obs, state_infos):
        """
        Return the distribution information about the actions.
        :param obs_var: observation values
        :param state_info_vars: a dictionary whose values should contain information about the state of the policy at
        the time it received the observation
        :return:
        """
        raise NotImplementedError
