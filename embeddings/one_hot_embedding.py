import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import ext
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.core import Parameterized
from garage.tf.distributions import DiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.spaces import Box

from sandbox.embed2learn.embeddings import StochasticEmbedding
from sandbox.embed2learn.tf.network_utils import mlp
from sandbox.embed2learn.tf.network_utils import parameter


class OneHotEmbedding(StochasticEmbedding, Parameterized, Serializable):
    def __init__(self, embedding_spec, name="OneHotEmbedding"):
        """
        :param embedding_spec:
        :return:
        """
        assert isinstance(embedding_spec.latent_space, Box)
        assert (embedding_spec.input_space.flat_dim <=
                embedding_spec.latent_space.flat_dim)
        StochasticEmbedding.__init__(self, embedding_spec)
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        self.name = name
        self._variable_scope = tf.variable_scope(
            self.name, reuse=tf.AUTO_REUSE)
        self._name_scope = tf.name_scope(self.name)

        # Build default graph
        with self._name_scope:
            # inputs
            self._input = self.input_space.new_tensor_variable(
                name="input", extra_dims=1)

            with tf.name_scope("default", values=[self._input]):
                # network
                latent_var, mean_var, std_param_var, dist = self._build_graph(
                    self._input)

            # outputs
            self._latent = tf.identity(latent_var, name="latent")
            self._latent_mean = tf.identity(mean_var, name="latent_mean")
            self._latent_std_param = tf.identity(std_param_var,
                                                 "latent_std_param")
            self._latent_distribution = dist

            # compiled functions
            with tf.variable_scope("f_dist"):
                self._f_dist = tensor_utils.compile_function(
                    inputs=[self._input],
                    outputs=[
                        self._latent, self._latent_mean, self._latent_std_param
                    ],
                )

    @property
    def input(self):
        return self._input

    @property
    def latent(self):
        return self._latent

    @property
    def latent_mean(self):
        return self._latent_mean

    @property
    def latent_std_param(self):
        return self._latent_std_param

    @property
    def inputs(self):
        return self._input

    @property
    def outputs(self):
        return (self._latent, self._latent_mean, self._latent_std_param,
                self._latent_distribution)

    def _build_graph(self, from_input):
        latent_dim = self.latent_space.flat_dim
        small = 1e-5

        with self._variable_scope:
            with tf.variable_scope("dist_params"):
                if self._std_share_network:
                    # mean and std networks share an MLP
                    b = np.concatenate(
                        [
                            np.zeros(latent_dim),
                            np.full(latent_dim, self._init_std_param)
                        ],
                        axis=0)
                    b = tf.constant_initializer(b)
                    # b = tf.truncated_normal_initializer(
                    #     mean=b, stddev=small)
                    mean_std_network = mlp(
                        with_input=from_input,
                        output_dim=latent_dim * 2,
                        hidden_sizes=self._hidden_sizes,
                        hidden_nonlinearity=self._hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        output_b_init=b,
                        name="mean_std_network")
                    with tf.variable_scope("mean_network"):
                        mean_network = mean_std_network[..., :latent_dim]
                    with tf.variable_scope("std_network"):
                        std_network = mean_std_network[..., latent_dim:]
                else:
                    # separate MLPs for mean and std networks
                    # mean network
                    mean_network = mlp(
                        with_input=from_input,
                        output_dim=latent_dim,
                        hidden_sizes=self._hidden_sizes,
                        hidden_nonlinearity=self._hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        name="mean_network")

                    # std network
                    if self._adaptive_std:
                        b = tf.constant_initializer(self._init_std_param)
                        # b = tf.truncated_normal_initializer(
                        #     mean=self._init_std_param, stddev=small)
                        std_network = mlp(
                            with_input=from_input,
                            output_dim=latent_dim,
                            hidden_sizes=self._std_hidden_sizes,
                            hidden_nonlinearity=self._std_hidden_nonlinearity,
                            output_nonlinearity=self._output_nonlinearity,
                            output_b_init=b,
                            name="std_network")
                    else:
                        p = tf.constant_initializer(self._init_std_param)
                        # p = tf.truncated_normal_initializer(
                        #     mean=self._init_std_param, stddev=small)
                        std_network = parameter(
                            with_input=from_input,
                            length=latent_dim,
                            initializer=p,
                            trainable=self._learn_std,
                            name="std_network")

                mean_var = mean_network
                std_param_var = std_network

                with tf.variable_scope("std_limits"):
                    if self._min_std_param:
                        std_param_var = tf.maximum(std_param_var,
                                                   self._min_std_param)
                    if self._max_std_param:
                        std_param_var = tf.minimum(std_param_var,
                                                   self._max_std_param)

            with tf.variable_scope("std_parameterization"):
                # build std_var with std parameterization
                if self._std_parameterization == "exp":
                    std_var = tf.exp(std_param_var)
                elif std_parameterization == "softplus":
                    std_var = tf.log(1. + tf.exp(std_param_var))
                else:
                    raise NotImplementedError

            dist = tf.contrib.distributions.MultivariateNormalDiag(
                mean_var, std_var)

            latent_var = dist.sample(seed=ext.get_seed())

            return latent_var, mean_var, std_param_var, dist

    @overrides
    def get_params_internal(self, **tags):
        if tags.get("trainable"):
            params = [v for v in tf.trainable_variables(scope=self.name)]
        else:
            params = [v for v in tf.global_variables(scope=self.name)]

        return params

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, input_var, state_info_vars=None, name=None):
        with tf.name_scope(name, "dist_info_sym",
                           [input_var, state_info_vars]):
            _, mean, log_std, _ = self._build_graph(input_var)

            return dict(mean=mean, log_std=log_std)

    def latent_sym(self, input_var, name=None):
        with tf.name_scope(name, "latent_sym", [input_var]):
            latent, _, _, _ = self._build_graph(input_var)

            return latent

    @overrides
    def get_latent(self, an_input):
        input_dim = self.input_space.flat_dim
        latent_dim = self.latent_space.flat_dim
        flat_in = self.input_space.flatten(an_input)
        right_pad = latent_dim - input_dim
        padded = np.pad(flat_in, (0, right_pad), "constant")
        return padded, dict(mean=padded, log_std=np.zeros_like(padded))

    def get_latents(self, inputs):
        input_dim = self.input_space.flat_dim
        latent_dim = self.latent_space.flat_dim
        flat_in = self.input_space.flatten(an_input)
        right_pad = latent_dim - input_dim
        padded = np.pad(flat_in, [(0, 0), (0, right_pad)], "constant")
        return padded, dict(mean=padded, log_std=np.zeros_like(padded))

    def log_likelihood(self, an_input, latent):
        flat_in = self.input_space.flatten(an_input)
        _, mean, log_std = [x[0] for x in self._f_dist([flat_in])]
        return self._dist.log_likelihood(latent,
                                         dict(mean=mean, log_std=log_std))

    def log_likelihoods(self, inputs, latents):
        flat_in = self.input_space.flatten_n(inputs)
        _, means, log_stds = self._f_dist(flat_in)
        return self._dist.log_likelihood(latents,
                                         dict(mean=means, log_std=log_stds))

    def log_likelihood_sym(self, input_var, latent_var, name=None):
        with tf.name_scope(name, "log_likelihood_sym",
                           [input_var, latent_var]):
            _, _, _, dist = self._build_graph(input_var)
            return dist.log_prob(latent_var)

    def entropy_sym(self, input_var, name=None):
        with tf.name_scope(name, "entropy_sym", [input_var]):
            return tf.constant(0, dtype=np.float32)
