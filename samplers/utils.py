import numpy as np

from rllab.misc import tensor_utils


# Given a numpy ndarray of shape (n, d_1, ..., d_k), returns a sliding-window
# view of the array with shape ((n - window_size), window_size, d_1, ..., d_k)
#
# Using `smear=True` will return an array of shape
# (n, window_size, d_1, ..., d_k), with the last `window_size` elements of the
# returned ndarray being repeats of the last element without smearing.
#
# Based on https://gist.github.com/codehacken/708f19ae746784cef6e68b037af65788
#
# TODO(gh/17): this should probably be in tensor_utils
def sliding_window(t, window, step_size, smear=False):
    if window > t.shape[0]:
        raise ValueError("`window` must be <= `t.shape[0]`")
    elif window == t.shape[0]:
        return np.stack([t] * window)

    # TODO(gh/19): this is broken for other step sizes. The problem may be with
    # the transpose trick
    if step_size != 1:
        raise NotImplementedError

    # The stride trick works only on the last dimension of an ndarray, so we
    # operate on the transpose, which reverses the dimensions of t.
    t_T = t.T

    shape = t_T.shape[:-1] + (t_T.shape[-1] - window + 1 - step_size, window)
    strides = t_T.strides + (t_T.strides[-1] * step_size, )
    t_T_win = np.lib.stride_tricks.as_strided(
        t_T, shape=shape, strides=strides)

    # t_T_win has shape (d_k, d_k-1, ..., (n - window_size), window_size)
    # To arrive at the final shape, we first transpose the result to arrive at
    # (window_size, (n - window_size), d_1, ..., d_k), then swap the firs two
    # axes
    t_win = np.swapaxes(t_T_win.T, 0, 1)

    # Optionally smear the last element to preserve the first dimension
    if smear:
        t_win = tensor_utils.pad_tensor(t_win, t.shape[0], mode='last')

    return t_win
