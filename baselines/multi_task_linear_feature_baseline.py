import numpy as np

from garage.misc.overrides import overrides
from garage.baselines import LinearFeatureBaseline


class MultiTaskLinearFeatureBaseline(LinearFeatureBaseline):

    @overrides
    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        z = np.clip(path["latents"], -10, 10)
        n = len(path["rewards"])
        an = np.arange(n).reshape(-1, 1) / 100.0
        return np.concatenate(
            [o, o**2, z, z**2, an, an**2, an**3,
             np.ones((n, 1))], axis=1)
