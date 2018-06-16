import tensorflow as tf


def mlp(with_input=None,
        output_dim=None,
        hidden_sizes=None,
        hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=None,
        output_b_init=tf.zeros_initializer(),
        name="mlp"):
    with tf.variable_scope(name):
        prev = with_input
        for i, h in enumerate(hidden_sizes):
            prev = tf.layers.dense(
                prev, h, activation=hidden_nonlinearity, name="fc{}".format(i))
        out = tf.layers.dense(
            prev,
            output_dim,
            activation=output_nonlinearity,
            bias_initializer=output_b_init,
            name="out")
        return out


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
