import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Distribution, FULLY_REPARAMETERIZED
from tensorflow.python.ops import array_ops, random_ops
from garage.misc.overrides import overrides


class DiagonalGaussian(Distribution):

    def __init__(
        self,
        means,
        log_stds,
        dim,
    ):
        """
        :param means: A tenor that stores the mean(s) of the distribution(s)
        :param log_stds: A tensor that store the log standard deviation(s) of the distribution(s)
        :param dim:
        """
        parameters = locals()

        # TODO check reparameterized..
        super(DiagonalGaussian, self).__init__(
            dtype=tf.float32,
            reparameterization_type=FULLY_REPARAMETERIZED,
            validate_args=True,
            allow_nan_stats=True,
            parameters=parameters,
        )

        self._means = means
        self._log_stds = log_stds

        self._dim = dim

    @overrides
    def kl_divergence(self, other, name="kl_divergence"):
        old_means = self.means
        old_log_stds = self.log_stds
        new_means = other.means
        new_log_stds = other.log_stds
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = tf.square(old_means - new_means) + \
                    tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    @overrides
    def log_prob(self, value, name="log_prob"):
        means = self.means
        log_stds = self.log_stds
        zs = (value - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, axis=-1) - \
               0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    @property
    def means(self):
        return self._means

    @property
    def log_stds(self):
        return self._log_stds

    @property
    def dim(self):
        return self._dim

    @property
    def dist_info_specs(self):
        return [("mean", (self.dim,)), ("log_std", (self.dim,))]

    @overrides
    def _batch_shape_tensor(self):
        return array_ops.broadcast_dynamic_shape(
            array_ops.shape(self._means),
            array_ops.shape(self._log_stds))

    def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
        sampled = random_ops.random_normal(
            shape=shape, mean=0., stddev=1., dtype=self._means.dtype, seed=seed)
        return sampled * tf.exp(self._log_stds) + self._means

