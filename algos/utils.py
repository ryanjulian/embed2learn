import tensorflow as tf

# TODO(gh/17): these should either be in garage.tf.misc.tensor_utils or
# replaced by the equivalents there.


def flatten_batch(t, name="flatten_batch"):
    with tf.variable_scope(name):
        shape = [-1] + list(t.shape[2:])
        return tf.reshape(t, shape)


def flatten_batch_dict(d, name="flatten_batch_dict"):
    with tf.variable_scope(name):
        d_flat = dict()
        for k, v in d.items():
            d_flat[k] = flatten_batch(v)
        return d_flat


def filter_valids(t, valid, name="filter_valids"):
    with tf.variable_scope("filter_valids"):
        return tf.boolean_mask(t, valid)


def filter_valids_dict(d, valid, name="filter_valids_dict"):
    with tf.variable_scope(name):
        d_valid = dict()
        for k, v in d.items():
            d_valid[k] = tf.boolean_mask(v, valid)
        return d_valid
