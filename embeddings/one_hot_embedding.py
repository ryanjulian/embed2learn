import numpy as np

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.misc import logger

from garage.tf.misc import tensor_utils
from garage.tf.spaces import Box

from sandbox.embed2learn.embeddings import Embedding


class OneHotEmbedding(Embedding, Serializable):
    def __init__(self, name, embedding_spec):
        """
        :param embedding_spec:
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(embedding_spec.input_space, Box)
        super(OneHotEmbedding, self).__init__(embedding_spec)

    def get_params_internal(self, **tags):
        return dict()

    @property
    def vectorized(self):
        return True

    @overrides
    def get_latent(self, an_input):
        in_flat = self.input_space.flatten(an_input)
        in_pad = tensor_utils.pad_tensor(in_flat, self.latent_space.flat_dim)
        latent = self.latent_space.unflatten(in_pad)
        return (latent, None)

    def get_latents(self, inputs):
        in_flat = self.input_space.flatten_n(inputs)
        in_pad = tensor_utils.pad_tensor_n(in_flat, self.latent_space.flat_dim)
        latents = self.latent_space.unflatten_n(in_pad)
        return (latents, None)
