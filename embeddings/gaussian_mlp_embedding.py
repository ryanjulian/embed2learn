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
from sandbox.embed2learn.tf.network_utils import two_headed_mlp

class GaussianMLPEmbedding(StochasticEmbedding, Parameterized, Serializable):
    def __init__(self,
                 embedding_spec,
                 name="GaussianMLPEmbedding",
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
                 std_parameterization='exp',
                 normalize=False):
        """
        :param embedding_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden
          layers
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
        assert isinstance(embedding_spec.latent_space, Box)
        StochasticEmbedding.__init__(self, embedding_spec)
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        if mean_network or std_network:
            raise NotImplementedError

        self.name = name
        self._variable_scope = tf.variable_scope(
            self.name, reuse=tf.AUTO_REUSE)
        self._name_scope = tf.name_scope(self.name)

        # TODO: eliminate
        self._dist = DiagonalGaussian(self.latent_space.flat_dim)

        # Network parameters
        self._hidden_sizes = hidden_sizes
        self._learn_std = learn_std
        self._init_std = init_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._mean_network = mean_network
        self._std_network = std_network
        self._std_parameterization = std_parameterization
        self._normalize = normalize

        if self._normalize:
            latent_dim = self.latent_space.flat_dim
            self._max_std = np.sqrt(1.0 / latent_dim)
            self._init_std = self._max_std / 2.0

        # Tranform std arguments to parameterized space
        self._init_std_param = None
        self._min_std_param = None
        self._max_std_param = None
        if self._std_parameterization == 'exp':
            self._init_std_param = np.log(self._init_std)
            if self._min_std:
                self._min_std_param = np.log(self._min_std)
            if self._max_std:
                self._max_std_param = np.log(self._max_std)
        elif self._std_parameterization == 'softplus':
            self._init_std_param = np.log(np.exp(self._init_std) - 1)
            if self._min_std:
                self._min_std_param = np.log(np.exp(self._min_std) - 1)
            if self._max_std:
                self._max_std_param = np.log(np.exp(self._max_std) - 1)
        else:
            raise NotImplementedError

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
                    mean_std_network = mlp(
                        with_input=from_input,
                        output_dim=latent_dim * 2,
                        hidden_sizes=self._hidden_sizes,
                        hidden_nonlinearity=self._hidden_nonlinearity,
                        output_nonlinearity=self._output_nonlinearity,
                        # hidden_w_init=tf.orthogonal_initializer(1.0),
                        # output_w_init=tf.orthogonal_initializer(1.0),
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
                elif self._std_parameterization == "softplus":
                    std_var = tf.log(1. + tf.exp(std_param_var))
                else:
                    raise NotImplementedError

            if self._normalize:
                mean_var = tf.nn.l2_normalize(mean_var)
                #std_var = tf.nn.l2_normalize(std_var)

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
        # flat_in = self.input_space.flatten(an_input)
        # mean, log_std = [x[0] for x in self._f_dist([flat_in])]
        # rnd = np.random.normal(size=mean.shape)
        # latent = rnd * np.exp(log_std) + mean
        # return latent, dict(mean=mean, log_std=log_std)
        flat_in = self.input_space.flatten(an_input)
        latent, mean, log_std = [x[0] for x in self._f_dist([flat_in])]
        return latent, dict(mean=mean, log_std=log_std)

    def get_latents(self, inputs):
        # flat_in = self.input_space.flatten_n(inputs)
        # means, log_stds = self._f_dist(flat_in)
        # rnd = np.random.normal(size=means.shape)
        # latents = rnd * np.exp(log_stds) + means
        # return latents, dict(mean=means, log_std=log_stds)
        flat_in = self.input_space.flatten_n(inputs)
        latents, means, log_stds = self._f_dist(flat_in)
        return latents, dict(mean=means, log_std=log_stds)

    def get_reparam_latent_sym(self,
                               input_var,
                               latent_var,
                               old_dist_info_vars,
                               name=None):
        """
        Given inputs, old latent outputs, and a distribution of old latent
        outputs, return a symbolically reparameterized representation of the
        inputs in terms of the embedding parameters
        :param in_var:
        :param latent_var:
        :param old_dist_info_vars:
        :return:
        """
        with tf.name_scope(name, "get_reparam_latent_sym",
                           [input_var, latent_var, old_dist_info_vars]):
            new_dist_info_vars = self.dist_info_sym(input_var, latent_var)
            new_mean_var, new_log_std_var = new_dist_info_vars[
                "mean"], new_dist_info_vars["log_std"]
            old_mean_var, old_log_std_var = old_dist_info_vars[
                "mean"], old_dist_info_vars["log_std"]
            epsilon_var = (latent_var - old_mean_var) / (
                tf.exp(old_log_std_var) + 1e-8)
            new_latent_var = new_mean_var + epsilon_var * tf.exp(
                new_log_std_var)
        return new_latent_var

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
            # dist_info = self.dist_info_sym(input_var, latent_var)
            # means_var, log_stds_var = dist_info['mean'], dist_info['log_std']
            # return self._dist.log_likelihood_sym(
            #     latent_var, dict(mean=means_var, log_std=log_stds_var))
            _, _, _, dist = self._build_graph(input_var)
            return dist.log_prob(latent_var)

    def entropy_sym(self, input_var, name=None):
        with tf.name_scope(name, "entropy_sym", [input_var]):
            _, _, _, dist = self._build_graph(input_var)
            return dist.entropy()

    def entropy_sym_sampled(self, dist_info_vars, name=None):
        with tf.name_scope(name, "entropy_sym_sampled", [dist_info_vars]):
            return self._dist.entropy_sym(dist_info_vars)

    def log_diagnostics(self):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AverageEmbeddingStd', np.mean(np.exp(log_stds)))
