import tensorflow as tf

from garage.core.serializable import Serializable
from sandbox.embed2learn.core.nn import mlp_feedforward_op


class MLP(Serializable):

    def __init__(self,
                 input_dim,
                 output_dim,
                 input_ph=None,
                 hidden_sizes=(128, 128),
                 activation_fn=tf.nn.relu,
                 output_nonlinearity=None,
                 output_b_init=None,
                 name='multi_layers_perceptrons'):
        super(MLP, self).__init__()
        Serializable.quick_init(self, locals())
        if input_ph is not None:
            self._input_ph = input_ph
        else:
            self._input_ph = tf.placeholder(
                tf.float32,
                shape=[None, input_dim],
                name='mlp_input'
            )
        self.layer_sizes = hidden_sizes+(output_dim,)
        self.activation_fn = activation_fn
        self.output_nonlinearity = output_nonlinearity
        self.name = name
        self.output_b_init = output_b_init
        self._output_op = self.feedforward(self.input_ph)

    def feedforward(self, input_ph, return_preactivations=False, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            preactivation = mlp_feedforward_op(
                inputs=[input_ph],
                layer_sizes=self.layer_sizes,
                activation_fn=self.activation_fn,
                output_nonlinearity=None,
                output_b_init=self.output_b_init,
            )
        if self.output_nonlinearity:
            output = self.output_nonlinearity(preactivation)
        else:
            output = preactivation

        if return_preactivations:
            return output, preactivation
        return output

    @property
    def input_ph(self):
        return self._input_ph

    @property
    def output_op(self):
        return self._output_op
