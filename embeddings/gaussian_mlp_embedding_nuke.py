import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.tf.spaces import Box
from sandbox.embed2learn.core.networks import MLP
from sandbox.embed2learn.distributions import DiagonalGaussian
from sandbox.embed2learn.embeddings import StochasticEmbedding


class GaussianMLPEmbedding(StochasticEmbedding, Serializable):

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
                 std_network:MLP=None,
                 std_parameterization='exp'):
        Serializable.quick_init(self, locals())

        assert isinstance(embedding_spec.latent_space, Box)
        self.name = name

        # Create mean network and std network
        with tf.variable_scope(name):
            in_dim = (embedding_spec.input_space.flat_dim)
            latent_dim = (embedding_spec.latent_space.flat_dim)

            if mean_network is None:
                if std_share_network:
                    if std_parameterization == "exp":
                        init_std_param = np.log(init_std)
                    elif std_parameterization == "softplus":
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError

                    init_b = init_std_param
                    mean_network = MLP(
                        input_dim=in_dim,
                        output_dim=2 * latent_dim,
                        hidden_sizes=hidden_sizes,
                        activation_fn=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                        output_b_init=init_b,
                        name="mean_network",
                    )
                    print(mean_network.output_op)
                    l_mean, l_std_param = tf.split(mean_network.output_op, 2, axis=1, name="mean_slice")
                    print(l_mean)
                else:
                    mean_network = MLP(
                        input_dim=in_dim,
                        output_dim=latent_dim,
                        hidden_sizes=hidden_sizes,
                        activation_fn=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                        name="mean_network",
                    )
                    l_mean = mean_network.output_op
            else:
                l_mean = mean_network.output_op

            self._mean_network = mean_network
            self._mean_input = self._mean_network.input_ph

            if std_network is not None:
                l_std_param = std_network.output_op
                self._std_input = std_network.input_ph
            else:
                if adaptive_std:
                    std_network = MLP(
                        input_dim=in_dim,
                        input_ph=self._mean_network.input_ph,
                        output_dim=latent_dim,
                        hidden_sizes=std_hidden_sizes,
                        activation_fn=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                        name="std_network",
                    )
                    l_std_param = std_network.output_op
                elif std_share_network:
                    pass
                else:
                    if std_parameterization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parameterization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    l_std_param = tf.get_variable('std', shape=[in_dim], initializer=tf.constant_initializer(init_std_param))

            self.std_parameterization = std_parameterization
            self.std_share_network = std_share_network
            if std_parameterization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parameterization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._l_mean = l_mean
            self._l_std_param = l_std_param
            print(l_mean, l_std_param)
            if max_std:
                log_std_limit = tf.constant(np.log(max_std), dtype=tf.float32)
                self._l_std_param = tf.minimum(self._l_std_param, log_std_limit, name="log_std_clip")
            self._dist = DiagonalGaussian(self._l_mean, self._l_std_param, latent_dim)

        super(GaussianMLPEmbedding, self).__init__(embedding_spec)

    def _get_feed_dict(self, inputs):
        feeds = {self._mean_input: inputs}
        return feeds

    def get_latent(self, an_input):
        feed_dict = self._get_feed_dict(an_input)
        sess = tf.get_default_session()
        latent = sess.run(self._dist.sample(), feed_dict=feed_dict)
        return latent

    def get_latents(self, inputs):
        feed_dict = self._get_feed_dict(inputs)
        sess = tf.get_default_session()
        latents = sess.run(self._dist.sample(), feed_dict=feed_dict)
        return latents

    def log_likelihood(self, an_input, latent):
        feed_dict = self._get_feed_dict(an_input)
        sess = tf.get_default_session()
        log_prob = sess.run(self._dist.log_prob(latent), feed_dict=feed_dict)
        return log_prob

    def log_likelihoods(self, inputs, latents):
        feed_dict = self._get_feed_dict(inputs)
        sess = tf.get_default_session()
        log_prob = sess.run(self._dist.log_prob(latents), feed_dict=feed_dict)
        return log_prob

    @property
    def vectorized(self):
        return True

    @property
    def distribution(self):
        return self._dist

    @property
    def input_ph(self):
        return self._mean_input

    def entropy(self, name=None):
        return self._dist.entropy(name)
