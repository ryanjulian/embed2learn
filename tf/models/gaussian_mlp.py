import tensorflow as tf
import tensorflow.contrib.distributions as tfd

from sandbox.embed2learn.tf.models import Model


class GaussianMLP(Model):
    """
    Models a multivariate Gaussian distribution with a diagonal covariance
    matrix using an MLP for the means and a simple parameter for the standard
    deviation

    Design details:
    * Initial mean is always ~0.0
    * Initial standard deviation is always 1.0
    * Standard deviation uses a softplus parameterization
    """

    def __init__(
        self,
        name="GaussianMLP",
        mean_hidden_sizes=(32, 32),
        mean_hidden_nonlinearity=tf.nn.tanh,
        mean_output_nonlinearity=None,
        std_trainable=True,
        min_std=None,
        max_std=None,
    ):
        """
        :param name: Name of model in computation graph
        :param mean_hidden_sizes: iterable of sizes for the fully-connected
          hidden layers in the mean MLP
        :param mean_hidden_nonlinearity: nonlinearity used for all hidden
          layers in the mean MLP
        :param mean_output_nonlinearity: nonlinearity for the output layer of
          the mean MLP
        :param std_trainable: Whether the standard deviation parameter is
          trainable
        :param std_min: Minimum value for the standard deviation, enforced
          using a softclip
        :param std_max: Maximum value for the standard deviation, enforced
          using a softclip
        """


