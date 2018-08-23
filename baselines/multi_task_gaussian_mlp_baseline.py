"""This module implements gaussian mlp baseline."""
import numpy as np

from garage.baselines import Baseline
from garage.core import Parameterized
from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.regressors import GaussianMLPRegressor


class MultiTaskGaussianMLPBaseline(Baseline, Parameterized):
    """A value function using gaussian mlp network."""

    def __init__(
            self,
            env_spec,
            extra_dims=0,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
    ):
        """
        Constructor.

        :param env_spec:
        :param subsample_factor:
        :param num_seq_inputs:
        :param regressor_args:
        """
        Serializable.quick_init(self, locals())
        super(MultiTaskGaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=((env_spec.observation_space.flat_dim + extra_dims) *
                         num_seq_inputs, ),
            output_dim=1,
            name="vf",
            use_trust_region=True,
            **regressor_args)

    @overrides
    def fit(self, paths):
        """Fit regressor based on paths."""
        observations = np.concatenate([p["observations"] for p in paths])
        tasks = np.concatenate([p["tasks"] for p in paths])
        latents = np.concatenate([p["latents"] for p in paths])
        trajectories = np.concatenate([p["trajectories"] for p in paths])
        aug_obs = np.concatenate([observations, tasks, latents], axis=1)
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(aug_obs, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        """Predict value based on paths."""
        inputs = np.concatenate(
            (path["observations"], path["tasks"], path["latents"]), axis=1)
        return self._regressor.predict(inputs, ).flatten()

    @overrides
    def get_param_values(self, **tags):
        """Get parameter values."""
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        """Set parameter values to val."""
        self._regressor.set_param_values(flattened_params, **tags)
