import tensorflow as tf

# TODO(gh/17): these should either be in garage.tf.misc.tensor_utils or
# replaced by the equivalents there.


def flatten_batch(t, name=None):
    with tf.name_scope(name, "flatten_batch", [t]):
        shape = [-1] + list(t.shape[2:])
        return tf.reshape(t, shape)


def flatten_batch_dict(d, name=None):
    with tf.name_scope(name, "flatten_batch_dict", [d]):
        d_flat = dict()
        for k, v in d.items():
            d_flat[k] = flatten_batch(v)
        return d_flat


def filter_valids(t, valid, name=None):
    with tf.name_scope(name, "filter_valids", [t, valid]):
        return tf.boolean_mask(t, valid)


def filter_valids_dict(d, valid, name=None):
    with tf.name_scope(name, "filter_valids_dict", [d, valid]):
        d_valid = dict()
        for k, v in d.items():
            d_valid[k] = tf.boolean_mask(v, valid)
        return d_valid
