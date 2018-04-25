import numpy as np
import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.core.serializable import Serializable

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.embed2learn.embeddings.base import StochasticEmbedding
from sandbox.embed2learn.embeddings.mlp_embedding import MLPEmbedding


class GaussianMLPEmbedding(MLPEmbedding, StochasticEmbedding):
    def __init__(self,
                 name,
                 embedding_spec,
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 mean_network=None,
                 std_network=None,
                 std_parameterization='exp'):
        """
        :param embedding_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable?
        :param init_std: Inital std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers
            for std
        :param min_std: whether to make sure that the std is at least some
            threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parameterization: how the std should be parameterized.
            There are a few options:
            -exp: the logarithm of the std will be stored, and applied an
                  exponential transformation
            -softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(embedding_spec.latent_space, Box)
        StochasticEmbedding.__init__(self,embedding_spec)

        with tf.variable_scope(name):

            #init mlp network
            MLPEmbedding.__init__(
                self,
                name=name,
                embedding_spec=embedding_spec,
                hidden_sizes=hidden_sizes,
                learn_std=learn_std,
                init_std=init_std,
                adaptive_std=adaptive_std,
                std_share_network=std_share_network,
                std_hidden_sizes=std_hidden_sizes,
                min_std=min_std,
                std_hidden_nonlinearity=std_hidden_nonlinearity,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                mean_network=mean_network,
                std_network=std_network,
                std_parameterization=std_parameterization)


            in_var = self.mean_network.input_layer.input_var
            dist_info_sym = self.dist_info_sym(
                self.mean_network.input_layer.input_var, dict())
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]
        
            if max_std is not None:
                # clip log_std
                log_std_limit = tf.constant(np.log(max_std), dtype=tf.float32)
                log_std_var = tf.minimum(log_std_var, log_std_limit, name="log_std_clip")

            self._f_dist = tensor_utils.compile_function(
                inputs=[in_var],
                outputs=[mean_var, log_std_var],
            )

    @overrides
    def get_latent(self, an_input):
        flat_in = self.input_space.flatten(an_input)
        mean, log_std = [x[0] for x in self._f_dist([flat_in])]
        rnd = np.random.normal(size=mean.shape)
        latent = rnd * np.exp(log_std) + mean
        return latent, dict(mean=mean, log_std=log_std)

    def get_latents(self, inputs):
        flat_in = self.input_space.flatten_n(inputs)
        means, log_stds = self._f_dist(flat_in)
        rnd = np.random.normal(size=means.shape)
        latents = rnd * np.exp(log_stds) + means
        return latents, dict(mean=means, log_std=log_stds)


