"""Base classes for TensorFlow models"""

import abc

from garage.tf.core import Parameterized


class Model(Parameterized, metaclass=abc.ABCMeta):
    """ Base class for TensorFlow models """

    def __init__(self):
        Parameterized.__init__(self)

    def inputs(self):
        pass

    def outputs(self):
        pass
