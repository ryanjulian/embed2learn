from sandbox.rocky.tf.core import Parameterized


class Embedding(Parameterized):
    def __init__(self, embedding_spec):
        Parameterized.__init__(self)
        self._embedding_spec = embedding_spec

    def get_latent(self, given):
        raise NotImplementedError

    def get_latents(self, givens):
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def vectorized(self):
        """
        Indicates whether the embedding is vectorized. If True, it should
        implement get_latents(), and support resetting with multiple
        simultaneous inputs.
        """
        return False

    @property
    def input_space(self):
        return self._embedding_spec.input_space

    @property
    def latent_space(self):
        return self._embedding_spec.latent_space

    @property
    def embedding_spec(self):
        return self._embedding_spec

    @property
    def recurrent(self):
        """
        Indicates whether the embedding is recurrent.
        :return:
        """
        return False

    def log_diagnostics(self):
        """
        Log extra information per iteration based on the collected paths
        """
        pass

    @property
    def state_info_keys(self):
        """
        Return keys for the information related to the embedding's state when
        taking it receives an input.
        :return:
        """
        return [k for k, _ in self.state_info_specs]

    @property
    def state_info_specs(self):
        """
        Return keys and shapes for the information related to the embedding's
        state when it receives an input.
        :return:
        """
        return list()

    def terminate(self):
        """
        Clean up operation
        """
        pass


class StochasticEmbedding(Embedding):
    @property
    def distribution(self):
        """
        :rtype Distribution
        """
        raise NotImplementedError

    def dist_info_sym(self, in_var, state_info_vars):
        """
        Return the symbolic distribution information about the latent variables.
        :param in_var: symbolic variable for input variables
        :param state_info_vars: a dictionary whose values should contain
            information about the state of the embedding at the time it received 
            the input variable
        :return:
        """
        raise NotImplementedError

    def dist_info(self, an_input, state_infos):
        """
        Return the distribution information about the latent variables.
        :param given: observation values
        :param state_info_vars: a dictionary whose values should contain 
            information about the state of the embedding at the time it received
            the input variable
        :return:
        """
        raise NotImplementedError
