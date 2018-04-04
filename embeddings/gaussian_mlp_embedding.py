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


class GaussianMLPEmbedding(StochasticEmbedding, LayersPowered, Serializable):
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

        with tf.variable_scope(name):
            in_dim = embedding_spec.input_space.flat_dim
            latent_dim = embedding_spec.latent_space.flat_dim

            # create network
            if mean_network is None:
                if std_share_network:
                    if std_parameterization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parameterization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    # TODO(gh/): this isn't really the right way to initialize
                    # the standard deviation
                    init_b = tf.constant_initializer(init_std_param)
                    mean_network = MLP(
                        name="mean_network",
                        input_shape=(in_dim, ),
                        output_dim=latent_dim * 2,
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                        output_b_init=init_b,
                    )
                    l_mean = L.SliceLayer(
                        mean_network.output_layer,
                        slice(latent_dim),
                        name="mean_slice")
                else:
                    mean_network = MLP(
                        name="mean_network",
                        input_shape=(in_dim, ),
                        output_dim=latent_dim,
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                    )
                    l_mean = mean_network.output_layer
            self._mean_network = mean_network

            in_var = mean_network.input_layer.input_var

            if std_network is not None:
                l_std_param = std_network.output_layer
            else:
                if adaptive_std:
                    std_network = MLP(
                        name="std_network",
                        input_shape=(in_dim, ),
                        input_layer=mean_network.input_layer,
                        output_dim=latent_dim,
                        hidden_sizes=std_hidden_sizes,
                        hidden_nonlinearity=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                    )
                    l_std_param = std_network.output_layer
                elif std_share_network:
                    l_std_param = L.SliceLayer(
                        mean_network.output_layer,
                        slice(latent_dim, 2 * latent_dim),
                        name="l_std_slice")
                else:
                    if std_parameterization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parameterization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    l_std_param = L.ParamLayer(
                        mean_network.input_layer,
                        num_units=latent_dim,
                        param=tf.constant_initializer(init_std_param),
                        name="output_std_param",
                        trainable=learn_std,
                    )

            self.std_parameterization = std_parameterization

            if std_parameterization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parameterization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._l_mean = l_mean
            self._l_std_param = l_std_param

            self._dist = DiagonalGaussian(latent_dim)

            LayersPowered.__init__(self, [l_mean, l_std_param])
            super(GaussianMLPEmbedding, self).__init__(embedding_spec)

            dist_info_sym = self.dist_info_sym(
                mean_network.input_layer.input_var, dict())
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            self._f_dist = tensor_utils.compile_function(
                inputs=[in_var],
                outputs=[mean_var, log_std_var],
            )

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, in_var, state_info_vars=None):
        mean_var, std_param_var = L.get_output(
            [self._l_mean, self._l_std_param], in_var)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parameterization == 'exp':
            log_std_var = std_param_var
        elif self.std_parameterization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        return dict(mean=mean_var, log_std=log_std_var)

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

    def get_reparam_latent_sym(self, in_var, latent_var, old_dist_info_vars):
        """
        Given inputs, old latent outputs, and a distribution of old latent
        outputs, return a symbolically reparameterized representation of the
        inputs in terms of the embedding parameters
        :param in_var:
        :param latent_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(in_var, latent_var)
        new_mean_var, new_log_std_var = new_dist_info_vars[
            "mean"], new_dist_info_varsl["og_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars[
            "mean"], old_dist_info_vars["log_std"]
        epsilon_var = (latent_var - old_mean_var) / (
            tf.exp(old_log_std_var) + 1e-8)
        new_latent_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_latent_var

    def log_likelihood(self, an_input, latent):
        flat_in = self.input_space.flatten(an_input)
        mean, log_std = [x[0] for x in self._f_dist([flat_in])]
        return self._dist.log_likelihood(latent,
                                         dict(mean=mean, log_std=log_std))

    def log_likelihoods(self, inputs, latents):
        flat_in = self.input_space.flatten_n(inputs)
        means, log_stds = self._f_dist(flat_in)
        return self._dist.log_likelihood(latents,
                                         dict(mean=means, log_std=log_stds))

    def log_likelihood_sym(self, input_var, latent_var):
        dist_info = self.dist_info_sym(input_var, latent_var)
        means_var, log_stds_var = dist_info['mean'], dist_info['log_std']
        return self._dist.log_likelihood_sym(latent_var,
                                             dict(
                                                 mean=means_var,
                                                 log_std=log_stds_var))

    def entropy(self, dist_info):
        return self.distribution.entropy(dist_info)

    def entropy_sym(self, dist_info_var):
        return self.distribution.entropy_sym(dist_info_var)

    def log_diagnostics(self):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AverageEmbeddingStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
