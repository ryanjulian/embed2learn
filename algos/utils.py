import tensorflow as tf

# TODO(gh/17): these should either be in sandbox.rocky.tf.misc.tensor_utils or
# replaced by the equivalents there.


def flatten_batch(t):
    shape = [-1] + list(t.shape[2:])
    return tf.reshape(t, shape)


def flatten_batch_dict(d):
    d_flat = dict()
    for k, v in d.items():
        d_flat[k] = flatten_batch(v)
    return d_flat


def filter_valids(t, valid):
    return tf.boolean_mask(t, valid)


def filter_valids_dict(d, valid):
    d_valid = dict()
    for k, v in d.items():
        d_valid[k] = tf.boolean_mask(v, valid)
    return d_valid
