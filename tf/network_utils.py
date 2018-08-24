import numpy as np
import tensorflow as tf


def softclip(x, min_, max_):
    offset = 0.5 * (min_ + max_)
    range_ = 0.5 * (max_ - min_)

    return offset + (range_ * tf.nn.tanh(x / range_))


def mlp(with_input=None,
        output_dim=None,
        hidden_sizes=None,
        hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=None,
        hidden_w_init=None,
        hidden_b_init=tf.zeros_initializer(),
        output_w_init=None,
        output_b_init=tf.zeros_initializer(),
        name="mlp"):
    with tf.variable_scope(name):
        prev = with_input
        for i, h in enumerate(hidden_sizes):
            prev = tf.layers.dense(
                prev, h,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init,
                name="fc{}".format(i)
            )
        out = tf.layers.dense(
            prev,
            output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="out")
        return out


def two_headed_mlp(with_input=None,
                   lower_output_dim=None,
                   upper_output_dim=None,
                   hidden_sizes=None,
                   hidden_nonlinearity=tf.nn.tanh,
                   lower_output_nonlinearity=None,
                   upper_output_nonlinearity=None,
                   hidden_w_init=None,
                   hidden_b_init=tf.zeros_initializer(),
                   lower_output_w_init=None,
                   lower_output_b_init=tf.zeros_initializer(),
                   upper_output_w_init=None,
                   upper_output_b_init=tf.zeros_initializer(),
                   name="mlp"):
    with tf.variable_scope(name):
        prev = with_input
        for i, h in enumerate(hidden_sizes):
            prev = tf.layers.dense(
                prev, h,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init,
                name="fc{}".format(i)
            )
        lower = tf.layers.dense(
            prev,
            lower_output_dim,
            activation=lower_output_nonlinearity,
            kernel_initializer=lower_output_w_init,
            bias_initializer=lower_output_b_init,
            name="lower")
        upper = tf.layers.dense(
            prev,
            upper_output_dim,
            activation=upper_output_nonlinearity,
            kernel_initializer=upper_output_w_init,
            bias_initializer=upper_output_b_init,
            name="upper")
        return lower, upper


def parameter(with_input,
              length,
              initializer=tf.zeros_initializer(),
              dtype=tf.float32,
              trainable=True,
              name="parameter"):
    with tf.variable_scope(name):
        p = tf.get_variable(
            "parameter",
            shape=(length, ),
            dtype=dtype,
            initializer=initializer,
            trainable=trainable)
        # TODO: this is ugly. There must be an idiomatic way to do it.
        ndim = with_input.get_shape().ndims
        reshaped_p = tf.reshape(p, (1, ) * (ndim - 1) + (length, ))
        tile_arg = tf.concat(
            axis=0, values=[tf.shape(with_input)[:ndim - 1], [1]])
        tiled = tf.tile(reshaped_p, tile_arg)
        return tiled
