from rllab.core import Serializable
from rllab.spaces import Space


class EmbeddingSpec(Serializable):
    def __init__(self, input_space, latent_space):
        """
        :type input_space: Space
        :type latent_space: Space
        """
        Serializable.quick_init(self, locals())
        self._input_space = input_space
        self._latent_space = latent_space

    @property
    def input_space(self):
        return self._input_space

    @property
    def latent_space(self):
        return self._latent_space
