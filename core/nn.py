import tensorflow as tf


def mlp_feedforward_op(inputs, layer_sizes, activation_fn=tf.nn.relu, output_nonlinearity=None, output_b_init=None):
    def bias(n_units, initializer=tf.zeros_initializer()):
        return tf.get_variable(
            name='bias', shape=n_units, initializer=initializer)

    def linear(x, n_units, postfix=None):
        input_size = x.shape[-1].value
        weight_name = 'weight' + '_' + str(postfix) if postfix else 'weight'
        weight = tf.get_variable(
            name=weight_name,
            shape=(input_size, n_units),
            initializer=tf.contrib.layers.xavier_initializer())

        return tf.tensordot(x, weight, axes=((-1, ), (0, )))

    out = 0
    for i, layer_size in enumerate(layer_sizes):
        with tf.variable_scope('layer_{i}'.format(i=i)):
            if i == 0:
                for j, input_tensor in enumerate(inputs):
                    out += linear(input_tensor, layer_size, j)
            else:
                out = linear(out, layer_size)

            if i == len(layer_sizes) - 1:
                if output_b_init:
                    out += bias(layer_size, initializer=tf.constant_initializer(output_b_init))
                else:
                    out += bias(layer_size)

            if i < len(layer_sizes) - 1 and activation_fn:
                out = activation_fn(out)

    if output_nonlinearity:
        out = output_nonlinearity(out)

    return out
