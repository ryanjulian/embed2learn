import numpy as np

from sandbox.rocky.tf.spaces.box import Box


def concat_spaces(top, bottom):
    assert isinstance(top, Box)
    assert isinstance(bottom, Box)

    top_lb, top_ub = top.bounds
    bottom_lb, bottom_ub = bottom.bounds
    return Box(
        np.concatenate([top_lb, bottom_lb]),
        np.concatenate([top_ub, bottom_ub]))
