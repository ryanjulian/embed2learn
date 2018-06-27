import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.tf.core import Parameterized
from garage.tf.spaces import Box
from sandbox.embed2learn.core.networks import MLP
from sandbox.embed2learn.distributions import DiagonalGaussian
from sandbox.embed2learn.embeddings import StochasticMultitaskPolicy


class GaussianMLPMultitaskPolicy(StochasticMultitaskPolicy, Parameterized, Serializable):

    def __init__(self,
                 env_spec,
                 embedding,
                 task_space,
                 name="GaussianMLPMultitaskPolicy",
                 sample_seed=1,
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 min_std=1e-6,
                 std_hidden_nonlinearity=tf.nn.tanh,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 mean_network=None,
                 std_network=None,
                 std_parameterization='exp',
                 n_itr=500):
        StochasticMultitaskPolicy.__init__(self, env_spec, embedding,
                                           task_space)
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)
        self.name = name

        super(GaussianMLPMultitaskPolicy, self).__init__(
            env_spec, embedding, task_space)

        with tf.variable_scope(name):
            task_obs_dim = self.task_observation_space.flat_dim
            action_dim = self.action_space.flat_dim
            latent_dim = self.latent_space.flat_dim
            obs_dim = self.observation_space.flat_dim

            latent_obs_dim = latent_dim + obs_dim

            self.task_input_ph = self._embedding.input_ph

            bp_latents = self._embedding.distribution.sample(name='latent_samples_bp', seed=sample_seed)
            bp_latents_tiled = tf.tile(tf.expand_dims(bp_latents, 1),
                                 [1, n_itr, 1])
            self.bp_latents = tf.reshape(bp_latents_tiled, shape=[-1, latent_dim])

            self.ff_latents = self._embedding.distribution.sample(name='latent_samples_ff', seed=sample_seed)
            self.ff_latents_ph = tf.placeholder(tf.float32, shape=[None, latent_dim], name="ff_latent_ph")
            self._obs_ph = tf.placeholder(tf.float32, shape=[None, obs_dim], name='obs_ph')

            ff_input_layer = tf.concat([self._obs_ph, self.ff_latents_ph], axis=1, name='ff_input_layer')
            input_layer = tf.concat([self._obs_ph, self.bp_latents], axis=1, name='bp_input_layer')

            self._ff_input = ff_input_layer
            self._bp_input = input_layer

            # Create the mean and std networks
            if mean_network is None:
                if std_share_network:
                    if std_parameterization == "exp":
                        init_std_param = np.log(init_std)
                    elif std_parameterization == "softplus":
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError

                    init_b = tf.constant_initializer(init_std_param)
                    mean_network = MLP(
                        input_dim=latent_obs_dim,
                        input_ph=input_layer,
                        output_dim=2 * latent_dim,
                        hidden_sizes=hidden_sizes,
                        activation_fn=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                        output_b_init=init_b,
                        name="mean_network",
                    )
                    l_mean = tf.slice(mean_network.output_op, 0, latent_dim, name="mean_slice")
                    ff_op = mean_network.feedforward(ff_input_layer, reuse=True)
                    self.ff_mean_op = tf.slice(ff_op, 0, latent_dim, name="ff_mean_slice")
                    self.ff_std_op = tf.slice(ff_op, latent_dim, latent_dim, name="ff_std_slice")
                else:
                    mean_network = MLP(
                        input_dim=latent_obs_dim,
                        input_ph=input_layer,
                        output_dim=latent_dim,
                        hidden_sizes=hidden_sizes,
                        activation_fn=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                        name="mean_network",
                    )
                    l_mean = mean_network.output_op
                    print(ff_input_layer)
                    self.ff_mean_op = mean_network.feedforward(ff_input_layer, reuse=True)
            else:
                l_mean = mean_network.output_op
                self.ff_mean_op = mean_network.feedforward(ff_input_layer, reuse=True)

            self._mean_network = mean_network

            if std_network is not None:
                l_std_param = std_network.output_op
                self._std_input = std_network.input_ph
            else:
                if adaptive_std:
                    std_network = MLP(
                        input_dim=latent_obs_dim,
                        input_ph=input_layer,
                        output_dim=latent_dim,
                        hidden_sizes=std_hidden_sizes,
                        activation_fn=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                        name="std_network",
                    )
                    l_std_param = std_network.output_op
                    self._std_input = std_network.input_ph
                    self.ff_std_op = std_network.feedforward(ff_input_layer, reuse=True)
                elif std_share_network:
                    l_std_param = tf.slice(mean_network.output_op, latent_dim, latent_dim, name="std_slice")
                else:
                    raise NotImplementedError
                    # if std_parameterization == 'exp':
                    #     init_std_param = np.log(init_std)
                    # elif std_parameterization == 'softplus':
                    #     init_std_param = np.log(np.exp(init_std) - 1)
                    # else:
                    #     raise NotImplementedError
                    # l_std_param = tf.get_variable('std', shape=[latent_dim], initializer=tf.constant_initializer(init_std_param))

            self.std_parameterization = std_parameterization
            self.std_share_network = std_share_network
            if std_parameterization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parameterization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._l_mean = l_mean
            self._l_std_param = l_std_param
            self._dist = DiagonalGaussian(l_mean, l_std_param, action_dim)
            self._ff_dist = DiagonalGaussian(self.ff_mean_op, self.ff_std_op, action_dim)
            self._get_action_op()

    @property
    def vectorized(self):
        return True

    @property
    def distribution(self):
        return self._dist

    @property
    def obs_ph(self):
        return self.obs_ph

    @property
    def task_ph(self):
        return self.embedding.input_ph

    @property
    def dist(self):
        return self._dist

    def get_latent(self, task_one_hot):
        sess = tf.get_default_session()
        sess.run(self.ff_latents, feed_dict={self.task_ph: task_one_hot})

    def _get_action_op(self):
        with tf.name_scope("get_action"):
            self._action_op = self._ff_dist.sample()

    def get_action(self, observation):

        flat_task_obs = self.task_observation_space.flatten(observation)
        flat_task, flat_obs = self.split_observation(flat_task_obs)

        sess = tf.get_default_session()
        feed_dict = {self.task_ph: flat_task}
        latent_info = sess.run([self.ff_latents, self._embedding.means, self._embedding.stds], feed_dict=feed_dict)

        feed_dict = {self.obs_ph: flat_obs, self.ff_latents_ph: latent_info}

        action, mean, std = sess.run([self._action_op, self.ff_mean_op, self.ff_std_op], feed_dict=feed_dict)


    def get_params_internal(self, **tags):

        if tags.get("trainable"):
            params = [v for v in tf.trainable_variables(scope=self.name)]
        else:
            params = [v for v in tf.global_variables(scope=self.name)]

        embed_params = self._embedding.get_params(**tags)
        params.extend(embed_params)
        return params


