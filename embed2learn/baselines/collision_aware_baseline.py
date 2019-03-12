import numpy as np

from garage.baselines import Baseline
from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core import Parameterized
from garage.tf.regressors import GaussianMLPRegressor


class CollisionAwareBaseline(Baseline, Parameterized):
    """A value function using gaussian mlp network."""

    def __init__(
            self,
            env_spec,
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
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())
        Baseline.__init__(self, env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=(
                (env_spec.observation_space.flat_dim + 1) * num_seq_inputs, ),
            output_dim=1,
            name="Baseline",
            **regressor_args)

    @overrides
    def fit(self, paths):
        """Fit regressor based on paths."""
        observations = np.concatenate([p["observations"] for p in paths])
        collisions = np.concatenate(np.float32(
            [p["env_infos"]["in_collision"] for p in paths]
        ))
        collisions = np.expand_dims(collisions, axis=1)
        aug_obs = np.concatenate([observations, collisions], axis=1)
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(aug_obs, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        """Predict value based on paths."""
        collisions = np.expand_dims(path["env_infos"]["in_collision"], axis=1)
        inputs = np.concatenate([path["observations"], collisions], axis=1)
        return self._regressor.predict(inputs, ).flatten()

    @overrides
    def get_param_values(self, **tags):
        """Get parameter values."""
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        """Set parameter values to val."""
        self._regressor.set_param_values(flattened_params, **tags)

    @overrides
    def get_params_internal(self, **tags):
        return self._regressor.get_params_internal(**tags)
