"""Package containing Models defined in TensorFlow."""

from sandbox.embed2learn.tf.models.base import Model
from sandbox.embed2learn.tf.models.gaussian_mlp import GaussianMLP

__all__ = ["Model", "GaussianMLP"]
