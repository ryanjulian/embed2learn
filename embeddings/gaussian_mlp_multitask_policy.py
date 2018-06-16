import itertools

import numpy as np
import tensorflow as tf

#from garage.core import Parameterized
from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.misc import logger

#from garage.tf.core import LayersPowered
#import garage.tf.core.layers as L
from garage.tf.core.network import MLP
from garage.tf.distributions import DiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.spaces import Box

from sandbox.embed2learn.embeddings import StochasticEmbedding
from sandbox.embed2learn.embeddings import StochasticMultitaskPolicy


class GaussianMLPMultitaskPolicy(StochasticMultitaskPolicy, Serializable):
    def __init__(self,
                 env_spec,
                 embedding,
                 task_space,
                 name="GaussianMLPMultitaskPolicy",
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
                 std_parametrization='exp'):
        """
        :param env_spec: observation space is a concatenation of task space and vanilla env observation space
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
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

            # task embedding + plain obs
            latent_obs_dim = latent_dim + obs_dim

            self.task_input = self._embedding._mean_network.input_layer
            self.task_input_var = self.task_input.input_var

            # self.env_input = L.InputLayer(
            #     (None, obs_dim), name="policy_env_input")
            # self.env_input_var = self.env_input.input_var
            self.env_input = self.observation_space.new_tensor_variable(
                name="env_input", extra_dims=1
            )

            embed_dist_info_sym = self._embedding.dist_info_sym(
                self.task_input.input_var)
            self.latent_mean_var = embed_dist_info_sym["mean"]
            self.latent_log_std_var = embed_dist_info_sym["log_std"]
            # self.latent = L.InputLayer(
            #     (None, latent_dim), self.latent_mean_var, name="latent_input")
            self.latent = self._embedding.latent_space.new_tensor_variable(
                name="latent_input",
                extra_dims=1
            )

            #self._policy_input = L.ConcatLayer((self.latent, self.env_input))
            self._policy_input = tf.concat(
                [self.latent, self.env_input], axis=1)

            # create network
            # if mean_network is None:
            #     mean_network = MLP(
            #         name="mean_network",
            #         input_shape=(latent_obs_dim, ),
            #         input_layer=self._policy_input,
            #         output_dim=action_dim,
            #         hidden_sizes=hidden_sizes,
            #         hidden_nonlinearity=hidden_nonlinearity,
            #         output_nonlinearity=output_nonlinearity,
            #     )
            #self._mean_network = mean_network

            # create network
            l_mean = None
            if mean_network is None:
                with tf.variable_scope("mean_network"):
                    prev = self._policy_input
                    for i, h in enumerate(hidden_sizes):
                        prev = tf.layers.dense(
                            prev,
                            h,
                            activation=hidden_nonlinearity,
                            name="fc{}".format(i))
                    out = tf.layers.dense(
                        prev,
                        action_dim,
                        activation=output_nonlinearity,
                        name="out")
                l_mean = out
            else:
                l_mean = mean_network

            # if std_network is not None:
            #     l_std_param = std_network.output_layer
            # else:
            #     if adaptive_std:
            #         std_network = MLP(
            #             name="std_network",
            #             input_shape=(latent_obs_dim, ),
            #             input_layer=mean_network.input_layer,
            #             output_dim=action_dim,
            #             hidden_sizes=std_hidden_sizes,
            #             hidden_nonlinearity=std_hidden_nonlinearity,
            #             output_nonlinearity=None,
            #         )
            #         l_std_param = std_network.output_layer
            #     else:
            #         if std_parametrization == 'exp':
            #             init_std_param = np.log(init_std)
            #         elif std_parametrization == 'softplus':
            #             init_std_param = np.log(np.exp(init_std) - 1)
            #         else:
            #             raise NotImplementedError
            #         l_std_param = L.ParamLayer(
            #             mean_network.input_layer,
            #             num_units=action_dim,
            #             param=tf.constant_initializer(init_std_param),
            #             name="output_std_param",
            #             trainable=learn_std,
            #         )

            if std_network is not None:
                l_std_param = std_network
            else:
                if adaptive_std:
                    with tf.variable_scope("std_network"):
                        prev = self._policy_input
                        for i, h in enumerate(std_hidden_sizes):
                            prev = tf.layers.dense(
                                prev,
                                h,
                                activation=std_hidden_nonlinearity,
                                name="fc{}".format(i))
                        out = tf.layers.dense(prev, action_dim, name="out")
                    l_std_param = out
                else:
                    if std_parametrization == 'exp':
                        init_std_param = np.log(init_std)
                    elif std_parametrization == 'softplus':
                        init_std_param = np.log(np.exp(init_std) - 1)
                    else:
                        raise NotImplementedError
                    l_std_param = tf.Variable(
                        initial_value=init_std_param,
                        expected_shape=(action_dim),
                        trainable=learn_std,
                        name="output_std_param")

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            # mean_var, log_std_var = L.get_output([l_mean, l_std_param])
            #
            # if self.min_std_param is not None:
            #     log_std_var = tf.maximum(log_std_var, np.log(min_std))
            #
            # self._mean_var, self._log_std_var = mean_var, log_std_var

            self._l_mean = l_mean
            self._l_std_param = l_std_param

            self._dist = DiagonalGaussian(action_dim)

            #LayersPowered.__init__(self, [l_mean, l_std_param])

            # dist_info_sym = self.dist_info_sym(
            #     {
            #         self.env_input.input_var: self.env_input.input_var,
            #         self.task_input.input_var: self.task_input.input_var
            #     }, dict())
            dist_info_sym = self.dist_info_sym(self._policy_input)
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            # self._task_obs_action_dist = tensor_utils.compile_function(
            #     inputs=[self.task_input_var, self.env_input.input_var],
            #     outputs=[
            #         mean_var, log_std_var, self.latent_mean_var,
            #         self.latent_log_std_var
            #     ],
            # )
            self._task_obs_action_dist = tensor_utils.compile_function(
                inputs=[self.task_input_var, self.env_input],
                outputs=[
                    mean_var, log_std_var, self.latent_mean_var,
                    self.latent_log_std_var
                ],
            )

            # self._latent_obs_action_dist = tensor_utils.compile_function(
            #     inputs=[self.latent_mean_var, self.env_input.input_var],
            #     outputs=[mean_var, log_std_var],
            # )
            self._latent_obs_action_dist = tensor_utils.compile_function(
                inputs=[self.latent_mean_var, self.env_input],
                outputs=[mean_var, log_std_var],
            )

    @property
    def vectorized(self):
        return True

    # @overrides
    # def get_params_internal(self, **tags):
    #     layers = L.get_all_layers(
    #         self._output_layers, treat_as_input=self._input_layers)
    #     layers += L.get_all_layers(
    #         self._embedding._output_layers,
    #         treat_as_input=self._embedding._input_layers)
    #     params = itertools.chain.from_iterable(
    #         l.get_params(**tags) for l in layers)
    #     return L.unique(params)

    @overrides
    def get_params(self, trainable=False):
        return tf.trainable_variables(scope=self.name)


    def dist_info_sym(self,
                      obs_var,
                      state_info_vars=None,
                      name="dist_info_sym"):
        with tensor_utils.enclosing_scope(self.name, name):
            # mean_var, std_param_var = L.get_output(
            #     [self._l_mean, self._l_std_param], obs_var)
            mean_var = self._l_mean
            std_param_var = self._l_std_param
            if self.min_std_param is not None:
                std_param_var = tf.maximum(std_param_var, self.min_std_param)
            if self.std_parametrization == 'exp':
                log_std_var = std_param_var
            elif self.std_parametrization == 'softplus':
                log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
            else:
                raise NotImplementedError
            return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        """

        :param observation: task onehot + env observation
        :return: action, dict
        """
        flat_task_obs = self.task_observation_space.flatten(observation)
        flat_task, flat_obs = self.split_observation(flat_task_obs)
        # evaluate embedding
        mean, log_std, latent_mean, latent_log_std = \
            [x[0] for x in self._task_obs_action_dist([flat_task], [flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        latent_info = dict(mean=latent_mean, log_std=latent_log_std)
        return action, dict(
            mean=mean, log_std=log_std, latent_info=latent_info)

    def get_actions(self, observations):
        # TODO implement split_observation_n(...)
        raise NotImplementedError()
        # flat_obs = self.task_observation_space.flatten_n(observations)
        # means, log_stds, latents = self._task_obs_action_dist(flat_obs)
        # rnd = np.random.normal(size=means.shape)
        # actions = rnd * np.exp(log_stds) + means
        # return actions, dict(mean=means, log_std=log_stds, latent=latents)

    @overrides
    def get_action_from_latent(self, observation, latent):
        flat_obs = self.observation_space.flatten(observation)
        flat_latent = self.latent_space.flatten(latent)
        # xs = self._latent_obs_action_dist(flat_latent, flat_obs)
        mean, log_std = [
            x[0]
            for x in self._latent_obs_action_dist([flat_latent], [flat_obs])
        ]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    # TODO implement these?
    def get_action_from_onehot(self, observation, onehot):
        raise NotImplementedError

    def get_actions_from_onehot(self, observations, onehots):
        raise NotImplementedError

    def get_actions_from_latent(self, observations, latents):
        raise NotImplementedError

    def get_reparam_action_sym(self,
                               obs_var,
                               action_var,
                               old_dist_info_vars,
                               name="get_reparam_action_sym"):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        with tensor_utils.enclosing_scope(self.name, name):
            new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
            new_mean_var, new_log_std_var = new_dist_info_vars[
                "mean"], new_dist_info_vars["log_std"]
            old_mean_var, old_log_std_var = old_dist_info_vars[
                "mean"], old_dist_info_vars["log_std"]
            epsilon_var = (action_var - old_mean_var) / (
                tf.exp(old_log_std_var) + 1e-8)
            new_action_var = new_mean_var + epsilon_var * tf.exp(
                new_log_std_var)
            return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
